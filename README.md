# MAKER Framework - Towers of Hanoi Implementation

This project implements the MAKER (Massively Decomposed Agentic Processes) framework from the paper "SOLVING A MILLION-STEP LLM TASK WITH ZERO ERRORS" ( [arXiv:2511.09030v1](https://arxiv.org/abs/2511.09030) ). It runs a 10-disk (1,023-step) Towers of Hanoi simulation using the MAKER framework to get the next move from an LLM at each step.

## Project Overview

The MAKER framework is designed to solve complex tasks by breaking them down into smaller, manageable steps and using LLM voting to ensure accuracy. This implementation specifically tests the framework on the classic Towers of Hanoi puzzle with 10 disks, requiring 1,023 moves to complete.

## Features

- **MAKER Framework Implementation**: Implements the first-to-ahead-by-k voting mechanism
- **Ground Truth Validation**: Uses an internal simulator to validate LLM moves
- **Red-Flagging Parser**: Parses and validates LLM responses with error handling
- **Comprehensive Error Handling**: Includes validation for moves and state consistency
- **Detailed Logging**: Provides step-by-step progress tracking

## Setup Instructions

### Prerequisites

1. `uv`
2. LLM model provider

### Installation

1. Install the required dependencies:

```bash
uv add openai python-dotenv
```

2. Set your OpenAI API key as an environment variable:
   - On macOS/Linux: `export OPENAI_API_KEY='your_api_key_here'`
   - On Windows: `set OPENAI_API_KEY=your_api_key_here`

   Or, you can hardcode it in the `API_KEY` variable in `main.py`, but this is not recommended for security.

## Usage

### Basic Usage

Run the simulation:
```bash
uv run main.py
```

### Configuration

The script can be configured through environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_API_BASE`: Custom OpenAI API base URL
- `LLM_MODEL`: LLM model to use
- `LLM_TEMPERATURE`: LLM temperature for decorrelation (default: 0.1)
- `N_DISKS`: Number of disks for Towers of Hanoi (default: 10)
- `K_THRESHOLD`: Vote margin required (default: 3)
- `MAX_OUTPUT_TOKENS`: Maximum output tokens for LLM responses (default: 750)

### Command Line Arguments

The script accepts one optional argument:
- `--disks N`: Set the number of disks ( overrides N_DISKS)

Example:
```bash
uv run main.py --disks 8
```

## Project Structure

```
.
├── main.py           # Main implementation of MAKER framework
├── README.md         # This file
├── .env              # Environment variables (not tracked in git)
├── .env.example      # Example environment variables
├── pyproject.toml    # Project configuration
├── uv.lock          # Dependency lock file
└── .python-version  # Python version specification
```

## How It Works

1. **Initialization**: Sets up the initial Towers of Hanoi state with all disks on peg 0
2. **Step-by-Step Simulation**: For each of the 1,023 steps:
   - Gets the ground truth move from the internal simulator
   - Runs the MAKER voting loop to get an LLM vote
   - Compares the LLM result with ground truth
   - Updates the state if they match
3. **Voting Mechanism**: Uses "first-to-ahead-by-k" voting where the first move that gets k votes ahead of the second-place move is selected
4. **Validation**: All LLM responses are parsed and validated using regex and AST parsing
5. **Error Handling**: If any error is detected, the simulation halts

## Key Components

### MAKER Framework Parameters
- `N_DISKS`: Number of disks (10 = 1,023 steps)
- `K_THRESHOLD`: Voting threshold (3)
- `LLM_TEMPERATURE`: Temperature for decorrelation (0.1)
- `MAX_OUTPUT_TOKENS`: Maximum output tokens (750)

### LLM Prompts
The system uses carefully crafted prompts directly from the paper's Appendix C, including:
- System prompt with clear instructions
- User template with rules and procedures
- Red-flagging parser for response validation

### Ground Truth Simulator
The internal simulator implements the exact Hanoi logic from the prompt:
- Rule 1: Move disk 1 clockwise if it didn't move last
- Rule 2: Make the only legal move that doesn't involve disk 1
- Comprehensive validation of all moves

## Performance

- **Total Steps**: 1,023 for 10 disks
- **Voting Samples**: Variable (stops when a move gets k votes ahead)
- **Timeout**: Controlled by OpenAI API limits
- **Memory**: Minimal (state represented as simple lists)

## Contributing

This is a research implementation. When contributing:

1. Maintain the existing code style and structure
2. Add comprehensive error handling
3. Include validation for all LLM inputs and outputs
4. Update documentation for new features
5. Test with various disk counts

## License

This project is part of research implementation. Please refer to the original paper "SOLVING A MILLION-STEP LLM TASK WITH ZERO ERRORS" for academic context.

## Acknowledgments

This implementation is based on the MAKER framework described in:
- "SOLVING A MILLION-STEP LLM TASK WITH ZERO ERRORS" (arXiv:2511.09030v1)
- Appendix C of the paper contains the exact prompts and validation logic used here
