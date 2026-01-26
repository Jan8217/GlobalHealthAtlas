"""
Scorer for GlobalHealthAtlas
"""
import argparse
import sys
import os
# Add the src directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.main import main as run_scoring


def score_data(input_file: str = None, output_file: str = None):
    """
    Run scoring on input data
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
    """
    if input_file and output_file:
        # TODO: Implement custom file processing if needed
        print(f"Scoring {input_file} -> {output_file}")
    else:
        # Run the main scoring process
        run_scoring()


def main():
    parser = argparse.ArgumentParser(description='GlobalHealthAtlas - Scoring Tool')
    parser.add_argument('--input-file', type=str, help='Input JSON file to process')
    parser.add_argument('--output-file', type=str, help='Output JSON file for results')
    
    args = parser.parse_args()
    
    score_data(args.input_file, args.output_file)


if __name__ == "__main__":
    main()