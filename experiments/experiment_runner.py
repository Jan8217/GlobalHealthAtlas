"""
Experiment runner for GlobalHealthAtlas
"""
import argparse
import sys
import os
# Add the src directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.main import main as run_scoring


def run_experiment():
    """Run experiments with the scoring system"""
    print("Running GlobalHealthAtlas experiments...")
    run_scoring()


def main():
    parser = argparse.ArgumentParser(description='GlobalHealthAtlas - Experiment Runner')
    parser.add_argument('--experiment-type', type=str, default='scoring', 
                        choices=['scoring', 'analysis'], help='Type of experiment to run')
    
    args = parser.parse_args()
    
    if args.experiment_type == 'scoring':
        run_scoring()
    elif args.experiment_type == 'analysis':
        from experiments.result_analyzer import analyze_results
        analyze_results()


if __name__ == "__main__":
    main()