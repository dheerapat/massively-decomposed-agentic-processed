"""
This script tests the MAKER (Massively Decomposed Agentic Processes)
framework from the paper "SOLVING A MILLION-STEP LLM TASK WITH ZERO ERRORS"
(arXiv:2511.09030v1).

It runs a 10-disk (1,023-step) Towers of Hanoi simulation, using the
MAKER framework to get the next move from an LLM at each step.

It validates the LLM's "voted" move against an internal ground-truth
simulator. The simulation will halt if an error is detected.

!!! SETUP INSTRUCTIONS !!!
1. Make sure you have the 'openai' package installed:
   pip install openai

2. Set your OpenAI API key as an environment variable.
   - On macOS/Linux: export OPENAI_API_KEY='your_api_key_here'
   - On Windows:       set OPENAI_API_KEY=your_api_key_here
   (Or, you can hardcode it in the `API_KEY` variable below,
    but this is not recommended for security.)
"""

import os
import re
import ast
import copy
import sys
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---

# !!! SET YOUR API KEY HERE if not using environment variables !!!
API_KEY = os.environ.get("OPENAI_API_KEY")
API_BASE = os.environ.get("OPENAI_API_BASE")

if not API_KEY:
    print("ERROR: OPENAI_API_KEY environment variable not set.")
    print("Please set the variable or hardcode it in the script.")
    sys.exit(1)

client = OpenAI(api_key=API_KEY, base_url=API_BASE)

# --- MAKER Framework Parameters ---
N_DISKS = 10  # 10 disks = 2^10 - 1 = 1,023 steps
K_THRESHOLD = 3  # Vote margin required (from the paper)
LLM_MODEL = os.environ.get("LLM_MODEL", "")
LLM_TEMPERATURE = 0.1  # For decorrelation (from the paper)
MAX_OUTPUT_TOKENS = 750  # Red-flagging for overly long responses

# --- LLM Prompts (Directly from paper Appendix C) ---

SYSTEM_PROMPT = """
You are a helpful assistant. Solve this puzzle for me.

There are three pegs and n disks of different sizes stacked on the first peg. The disks are
numbered from 1 (smallest) to n (largest). Disk moves in this puzzle should follow:
1. Only one disk can be moved at a time.
2. Each move consists of taking the upper disk from one stack and placing it on top of
another stack.
3. A larger disk may not be placed on top of a smaller disk.

The goal is to move the entire stack to the third peg.

Example: With 3 disks numbered 1 (smallest), 2, and 3 (largest), the initial state is [[3, 2,
1], [], []], and a solution might be:
moves = [[1, 0, 2], [2, 0, 1], [1, 2, 1], [3, 0, 2], [1, 1, 0], [2, 1, 2], [1, 0, 2]]
This means: Move disk 1 from peg 0 to peg 2, then move disk 2 from peg 0 to peg 1, and so on.

Requirements:
The positions are 0-indexed (the leftmost peg is 0).
Ensure your answer includes a single next move in this EXACT FORMAT:
"move = [disk id, from peg, to peg]"
Ensure your answer includes the next state resulting from applying the move to the current
state in this EXACT FORMAT:
"next_state = [[...], [...], [...]]"
"""

USER_TEMPLATE = """
Rules:
Only one disk can be moved at a time.
Only the top disk from any stack can be moved.
A larger disk may not be placed on top of a smaller disk.

For all moves, follow the standard Tower of Hanoi procedure:
If the previous move did not move disk 1, move disk 1 clockwise one peg (0->1->2->0).
If the previous move did move disk 1, make the only legal move that does not involve moving
disk 1.

Use these clear steps to find the next move given the previous move and current state.

Previous move: {previous_move}
Current State: {current_state}

Based on the previous move and current state, find the single next move that follows the
procedure and the resulting next state.
"""

# --- Red-Flagging Parser (Adapted from paper Appendix C) ---

def _validate_move(move):
    """Validates the 'move' list."""
    if not (isinstance(move, list) and
            len(move) == 3 and
            all(isinstance(x, int) for x in move)):
        raise ValueError("'move' must be a list of exactly 3 integers.")
    return move

def _validate_state(state, n_disks):
    """Validates the 'next_state' list."""
    if not (isinstance(state, list) and
            len(state) == 3 and
            all(isinstance(t, list) for t in state)):
        raise ValueError("'next_state' must be a list of three lists.")

    flat = []
    for peg in state:
        if not all(isinstance(x, int) for x in peg):
            raise ValueError("All entries in 'next_state' pegs must be integers.")
        # Check peg rule: larger disk never on smaller disk
        for i in range(len(peg) - 1):
            if peg[i] < peg[i+1]:
                raise ValueError(f"State rule violation: Peg {peg} has larger disk on smaller.")
        flat.extend(peg)

    # Check that all disks are present exactly once
    # This was modified from the paper's hardcoded '20' to use n_disks
    expected_disks = set(range(1, n_disks + 1))
    if len(flat) != n_disks or set(flat) != expected_disks:
        missing = sorted(expected_disks - set(flat))
        extra = sorted(set(flat) - expected_disks)
        raise ValueError(
            f"State must contain 1..{n_disks} exactly once. "
            f"Missing: {missing or '[]'}, Extra: {extra or '[]'}"
        )
    return state

def parse_move_state_flag(response_text: str, n_disks: int):
    """
    Parses and validates the LLM response.
    This is the "Red-Flagging" component.
    Raises ValueError if parsing or validation fails.
    """
    # Use regex to find the *last* occurrence of move and next_state
    move_pat = re.compile(r"(?is)\bmove\b\s*=\s*(\[[^\[\]]*\])")
    state_pat = re.compile(
        r"(?is)\bnext_state\b\s*=\s*(\[\s*\[[^\]]*\]\s*,\s*\[[^\]]*\]\s*,\s*\[[^\]]*\]\s*\])"
    )

    move_matches = list(move_pat.finditer(response_text))
    if not move_matches:
        raise ValueError("Red Flag: No 'move = [...]' found.")
    move_str = move_matches[-1].group(1)

    state_matches = list(state_pat.finditer(response_text))
    if not state_matches:
        raise ValueError("Red Flag: No 'next_state = [[...], [...], [...]]' found.")
    state_str = state_matches[-1].group(1)

    try:
        move = ast.literal_eval(move_str)
    except Exception as e:
        raise ValueError(f"Red Flag: Could not parse 'move' as a Python list. {e}")

    try:
        next_state = ast.literal_eval(state_str)
    except Exception as e:
        raise ValueError(f"Red Flag: Could not parse 'next_state' as Python lists. {e}")

    # Run validation checks
    valid_move = _validate_move(move)
    valid_state = _validate_state(next_state, n_disks)

    # Return canonical representations for voting
    return (tuple(valid_move), str(valid_state))

# --- Ground Truth Simulator ---

def get_ground_truth_move(state, prev_move, n_disks):
    """
    Implements the *exact* Hanoi logic provided in the LLM prompt.
    This serves as our infallible ground truth.
    """
    # Rule 1: If prev_move did not move disk 1, move disk 1 clockwise.
    if prev_move is None or prev_move[0] != 1:
        # Find disk 1
        disk_1_peg = -1
        for i, peg in enumerate(state):
            if peg and peg[-1] == 1:
                disk_1_peg = i
                break
        
        # Determine target peg (clockwise)
        target_peg = (disk_1_peg + 1) % 3
        return [1, disk_1_peg, target_peg]

    # Rule 2: If prev_move did move disk 1, make the only legal move
    #         that does not involve moving disk 1.
    else:
        possible_moves = []
        for from_peg in range(3):
            if not state[from_peg]:  # Can't move from empty peg
                continue
            
            disk_to_move = state[from_peg][-1]
            if disk_to_move == 1:  # Can't move disk 1
                continue

            for to_peg in range(3):
                if from_peg == to_peg:
                    continue
                
                # Check if move is legal
                if not state[to_peg]:  # Moving to empty peg is always legal
                    possible_moves.append([disk_to_move, from_peg, to_peg])
                elif disk_to_move < state[to_peg][-1]: # Moving to non-empty peg
                    possible_moves.append([disk_to_move, from_peg, to_peg])

        # According to the algorithm, there should be exactly one such move
        if len(possible_moves) != 1:
            raise Exception(
                f"Ground Truth Error: Expected 1 legal move, found {len(possible_moves)}. "
                f"State: {state}, Prev: {prev_move}"
            )
        return possible_moves[0]

def apply_move(state, move):
    """Applies a move to a state and returns the new state."""
    new_state = copy.deepcopy(state)
    disk, from_peg, to_peg = move
    
    if not new_state[from_peg] or new_state[from_peg][-1] != disk:
        raise Exception(f"Invalid move: Disk {disk} not on top of peg {from_peg}")
    
    popped_disk = new_state[from_peg].pop()
    
    if new_state[to_peg] and new_state[to_peg][-1] < popped_disk:
        raise Exception(f"Invalid move: Cannot place {popped_disk} on {new_state[to_peg][-1]}")
        
    new_state[to_peg].append(popped_disk)
    return new_state

# --- MAKER Core Logic ---

def get_llm_vote(current_state_str, prev_move_str):
    """Makes a single call to the OpenAI API."""
    try:
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(n=N_DISKS)},
                {"role": "user", "content": USER_TEMPLATE.format(
                    previous_move=prev_move_str,
                    current_state=current_state_str
                )}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"  [API Error: {e}]")
        return None  # Treat API errors as failed votes

def check_for_winner(vote_tally, k):
    """Checks if any candidate is 'k' votes ahead."""
    if not vote_tally:
        return None

    # Sort votes by count, descending
    sorted_votes = sorted(vote_tally.items(), key=lambda item: item[1], reverse=True)
    
    winner_key, winner_count = sorted_votes[0]
    
    second_place_count = 0
    if len(sorted_votes) > 1:
        second_place_count = sorted_votes[1][1]
        
    if winner_count >= k + second_place_count:
        return winner_key  # Returns (move_tuple, state_str)
    
    return None

def run_voting_loop(current_state, prev_move, n_disks, k):
    """
    Runs the 'First-to-ahead-by-k' voting loop for a single step.
    """
    vote_tally = {}
    total_samples = 0
    
    current_state_str = str(current_state)
    prev_move_str = str(prev_move)
    
    while True:
        total_samples += 1
        print(f"\r  Sampling... (Total samples: {total_samples})  ", end="")
        
        # 1. Get LLM response
        raw_response = get_llm_vote(current_state_str, prev_move_str)
        
        if raw_response is None:
            continue # Skip on API error

        # 2. Red-Flagging: Parse and Validate
        try:
            (move_key, state_key) = parse_move_state_flag(raw_response, n_disks)
            
            # 3. Tally Vote
            vote_key = (move_key, state_key)
            vote_tally[vote_key] = vote_tally.get(vote_key, 0) + 1
            
        except ValueError as e:
            # Red-flagged! Discard this vote.
            print(f"\n  [Red Flag: {e}]") # Uncomment for verbose logging
            pass
        
        # 4. Check for Winner
        winner = check_for_winner(vote_tally, k)
        if winner:
            print(f"\r  Winner found after {total_samples} samples. {vote_tally}")
            return winner # (move_tuple, state_str)

# --- Main Simulation ---

def main_simulation():
    print(f"Starting MAKER Framework Test: Towers of Hanoi")
    print(f"Disks: {N_DISKS}")
    print(f"Total Steps: {(2**N_DISKS) - 1}")
    print(f"Voting Threshold (k): {K_THRESHOLD}")
    print(f"LLM Model: {LLM_MODEL}\n")
    
    # Initialize simulation state
    total_steps = (2**N_DISKS) - 1
    current_state = [list(range(N_DISKS, 0, -1)), [], []]
    prev_move = None
    
    start_time = time.time()
    
    for step in range(1, total_steps + 1):
        print(f"--- Step {step}/{total_steps} ---")
        
        # 1. Get Ground Truth (for validation)
        try:
            correct_move = get_ground_truth_move(current_state, prev_move, N_DISKS)
            correct_next_state = apply_move(current_state, correct_move)
            correct_key = (tuple(correct_move), str(correct_next_state))
        except Exception as e:
            print(f"\n!!! SIMULATOR ERROR: {e} !!!")
            print("Halting simulation.")
            break
            
        # 2. Run MAKER Voting Loop
        (voted_move_tuple, voted_state_str) = run_voting_loop(
            current_state, prev_move, N_DISKS, K_THRESHOLD
        )
        voted_key = (voted_move_tuple, voted_state_str)
        
        # 3. Compare LLM result to Ground Truth
        if voted_key == correct_key:
            print(f"  SUCCESS: Voted move matches ground truth.")
            # Update state for next loop
            current_state = correct_next_state
            prev_move = correct_move
        else:
            print(f"\n!!! ERROR: LLM VOTED INCORRECTLY at Step {step} !!!\n")
            print(f"  Ground Truth Move:  {correct_key[0]}")
            print(f"  LLM Voted Move:     {voted_key[0]}\n")
            print(f"  Ground Truth State: {correct_key[1]}")
            print(f"  LLM Voted State:    {voted_key[1]}\n")
            print("Halting simulation.")
            break
    
    else:
        # This 'else' block runs only if the 'for' loop completes without 'break'
        print("\n\n" + "="*50)
        print("    SIMULATION COMPLETED WITH ZERO ERRORS!    ")
        print(f"    Successfully finished all {total_steps} steps.    ")
        print("="*50)

    end_time = time.time()
    print(f"\nTotal simulation time: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main_simulation()