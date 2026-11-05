from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGES = ('benchmark', 'learning', 'preprocessing', 'stability', 'controls')

def main():
    parser = argparse.ArgumentParser(description='Run the five point-prediction stages into a new local results directory')
    parser.add_argument('--profile', choices=['quick', 'full'], default='full')
    parser.add_argument('--stage', choices=['prepare', 'all', 'summarize'] + list(STAGES), default='all')
    parser.add_argument('--out')
    parser.add_argument('--max-tasks', type=int)
    args = parser.parse_args()
    out = Path(args.out).resolve() if args.out else ROOT / 'runs' / ('baseline_' + args.profile)
    archive = (ROOT / 'results').resolve()
    if out == archive or archive in out.parents:
        raise ValueError('recorded results are immutable; choose an output in runs or another local folder')
    sys.path.insert(0, str(ROOT / 'executed'))
    import run
    import pandas as pd
    original_plan = run.task_plan
    run.task_plan = lambda cfg, splits: [t for t in original_plan(cfg, splits) if t['stage'] in STAGES]
    original_signature = run.execution_signature
    def signature(cfg):
        ignored, context = original_signature(cfg)
        context['point_release_extension'] = dict(entrypoint_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), included_stages=list(STAGES), excludes_completed_historical_interval_stage=True)
        return run.payload_hash(context), context
    run.execution_signature = signature
    original_csv = pd.read_csv
    def read_csv(*positional, **keyword):
        keyword.setdefault('float_precision', 'round_trip')
        return original_csv(*positional, **keyword)
    pd.read_csv = read_csv
    sys.argv = ['run.py', '--config', str(ROOT / 'configs' / ('baseline_' + args.profile + '.json')), '--out', str(out), '--stage', args.stage]
    if args.max_tasks is not None:
        sys.argv += ['--max-tasks', str(args.max_tasks)]
    try:
        run.main()
    finally:
        pd.read_csv = original_csv

if __name__ == '__main__':
    main()
