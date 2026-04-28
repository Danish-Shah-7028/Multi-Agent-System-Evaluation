#!/usr/bin/env python3
"""
Main entry point for running the research study.
"""

import sys
import os
import argparse
import json
import logging
from pathlib import Path

# Load .env file at startup
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.experiment_runner import ExperimentRunner

def main():
    parser = argparse.ArgumentParser(
        description="Run Multi-Agent vs Single-Agent LLM Research Study"
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=None,
        help="Limit number of tasks to run (for testing). Default: all"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/tasks_dataset.json",
        help="Path to task dataset"
    )
    parser.add_argument(
        "--clear-memory",
        action="store_true",
        help="Clear System C memory before starting"
    )
    
    args = parser.parse_args()
    
    # Verify Groq API key(s)
    keys_1_4 = [os.environ.get(f"GROQ_API_KEY_{i}") for i in range(1, 5)]
    single_key = os.environ.get("GROQ_API_KEY")
    
    has_numbered_keys = any(keys_1_4)
    has_single_key = bool(single_key)
    
    if not has_numbered_keys and not has_single_key:
        print("ERROR: No Groq API keys found!")
        print("Set credentials using one of these formats:")
        print("  Option 1 (single key): GROQ_API_KEY=your_key_here")
        print("  Option 2 (multiple keys): GROQ_API_KEY_1=key1, GROQ_API_KEY_2=key2, etc.")
        print("\nOr add to .env file in the project root.")
        sys.exit(1)
    
    if has_numbered_keys:
        key_count = sum(1 for k in keys_1_4 if k)
        print(f"Using {key_count} Groq API key(s) from GROQ_API_KEY_1..4")
    else:
        print("Using single Groq API key from GROQ_API_KEY")
    
    # Initialize experiment runner
    print(f"Initializing experiment runner...")
    runner = ExperimentRunner(config_path=args.config, data_path=args.data)
    
    # Clear memory if requested
    if args.clear_memory:
        print("Clearing System C memory...")
        runner.clear_memory()
    
    # Run experiments
    print(f"Starting experiments on {len(runner.tasks)} tasks...")
    if args.tasks:
        print(f"Limiting to {args.tasks} tasks for testing")
    
    summary = runner.run_all_experiments(task_limit=args.tasks)
    
    # Display summary
    print("EXPERIMENT SUMMARY")
    print(f"Total experiments: {summary['total_experiments']}")
    print(f"Total tasks: {summary['total_tasks']}")
    print()
    
    for system, stats in summary['by_system'].items():
        print(f"\n{stats['name']}:")
        print(f"  Success Rate: {stats['success_rate']:.1f}% ({stats['successes']}/{stats['total_tasks']})")
        print(f"  Avg Steps: {stats['avg_steps']}")
        print(f"  Avg Tokens: {stats['avg_tokens']}")
        print(f"  Total Tokens: {stats['total_tokens']}")
        print(f"  Avg Similarity: {stats['avg_similarity']:.3f}")
    
    print('\n')
    
    # Save results
    print("Saving results...")
    runner.save_results()
    
    print("\nResults saved to results/ directory")
    print("View results/experiment_results.csv for detailed results")
    print("View results/experiment.log for full execution log")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
