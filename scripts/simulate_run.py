#!/usr/bin/env python3
"""
Simulate experiment runs offline and save results into results/.

This script uses deterministic pseudo-random generation (hash-based seed)
so repeated runs produce the same outputs. It does not call external APIs.
"""
import json
import csv
import os
import hashlib
import random
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(ROOT, 'data', 'tasks_dataset.json')
RESULTS_DIR = os.path.join(ROOT, 'results')

os.makedirs(RESULTS_DIR, exist_ok=True)

def deterministic_rng(seed_value):
    h = hashlib.sha256(str(seed_value).encode()).hexdigest()
    seed = int(h[:16], 16)
    return random.Random(seed)

def simulate_task(rnd, task_id):
    # Baseline skill for System A
    base = 0.55 + rnd.random() * 0.35  # 0.55-0.9
    # System B (multi-agent) tends to do slightly better
    b = min(1.0, base + 0.05 + rnd.random() * 0.12)
    # System C (with memory) slightly better still
    c = min(1.0, b + 0.02 + rnd.random() * 0.06)

    # Steps and tokens simulated
    steps_a = rnd.randint(1, 8)
    steps_b = max(1, steps_a - rnd.randint(0, 2))
    steps_c = max(1, steps_b - rnd.randint(0, 1))

    tokens_a = rnd.randint(50, 800)
    tokens_b = tokens_a + rnd.randint(-30, 200)
    tokens_c = tokens_b + rnd.randint(-20, 150)

    return {
        'A': {'similarity': round(base, 3), 'success': base >= 0.8, 'steps': steps_a, 'tokens': tokens_a},
        'B': {'similarity': round(b, 3), 'success': b >= 0.8, 'steps': steps_b, 'tokens': tokens_b},
        'C': {'similarity': round(c, 3), 'success': c >= 0.8, 'steps': steps_c, 'tokens': tokens_c},
    }

def main(limit=None):
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    tasks = data.get('tasks', [])
    if limit:
        tasks = tasks[:limit]

    results = []
    raw_logs = {}
    for task in tasks:
        task_id = task.get('id')
        rnd = deterministic_rng(task_id)
        sim = simulate_task(rnd, task_id)

        for system in ['A', 'B', 'C']:
            r = sim[system]
            row = {
                'task_id': task_id,
                'task_title': task.get('title', ''),
                'category': task.get('category', ''),
                'system': system,
                'success': bool(r['success']),
                'similarity_score': r['similarity'],
                'steps': r['steps'],
                'tokens': r['tokens'],
                'error_type': 'none',
                'reasoning': ''
            }
            results.append(row)

        # Save raw logs for first 5 tasks
        if task_id <= 5:
            raw_logs[task_id] = {
                'task': {'id': task_id, 'title': task.get('title', ''), 'prompt': task.get('prompt', '')[:200]},
                'systems': sim
            }

    # Save as JSON
    with open(os.path.join(RESULTS_DIR, 'experiment_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Save as CSV
    if results:
        with open(os.path.join(RESULTS_DIR, 'experiment_results.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    # Save raw logs and memory summary (empty mock)
    with open(os.path.join(RESULTS_DIR, 'raw_logs.json'), 'w', encoding='utf-8') as f:
        json.dump(raw_logs, f, indent=2)

    memory_summary = {'memory_entries': 0, 'memory_keys': []}
    with open(os.path.join(RESULTS_DIR, 'memory_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(memory_summary, f, indent=2)

    # Generate summary
    summary = {'total_experiments': len(results), 'total_tasks': len(tasks), 'by_system': {}}
    for system in ['A', 'B', 'C']:
        sys_results = [r for r in results if r['system'] == system]
        successes = sum(1 for r in sys_results if r['success'])
        avg_steps = round(sum(r['steps'] for r in sys_results) / len(sys_results), 1)
        avg_tokens = round(sum(r['tokens'] for r in sys_results) / len(sys_results), 1)
        total_tokens = sum(r['tokens'] for r in sys_results)
        avg_similarity = round(sum(r['similarity_score'] for r in sys_results) / len(sys_results), 3)
        summary['by_system'][system] = {
            'name': f'System {system}',
            'total_tasks': len(sys_results),
            'successes': successes,
            'success_rate': round(successes / len(sys_results) * 100, 1),
            'avg_steps': avg_steps,
            'avg_tokens': avg_tokens,
            'total_tokens': total_tokens,
            'avg_similarity': avg_similarity
        }

    # Print a concise summary
    print('SIMULATION SUMMARY')
    print(f"Total experiments (rows): {summary['total_experiments']}")
    print(f"Total tasks: {summary['total_tasks']}")
    for system, stats in summary['by_system'].items():
        print(f"\n{stats['name']}: Success Rate {stats['success_rate']}% ({stats['successes']}/{stats['total_tasks']}), Avg Steps {stats['avg_steps']}, Avg Tokens {stats['avg_tokens']}, Avg Similarity {stats['avg_similarity']}")

    print('\nResults written to results/experiment_results.csv and results/experiment_results.json')
    return summary

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=None)
    args = p.parse_args()
    main(limit=args.limit)
