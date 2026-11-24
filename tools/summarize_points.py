from pathlib import Path
import argparse
import json
import sys
import pandas as pd
from materialize_results import materialize

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description='Rebuild all point-prediction tables from recorded predictions and task caches without refitting models')
    parser.add_argument('--out', default=str(ROOT / 'work' / 'rebuilt_results'))
    args = parser.parse_args()
    out = materialize(args.out)
    sys.path.insert(0, str(ROOT / 'executed'))
    from summarize import summarize
    cfg = json.loads((ROOT / 'executed' / 'configs' / 'full.json').read_text(encoding='utf-8'))
    cfg['summary_make_figures'] = False
    original_csv = pd.read_csv
    def read_csv(*positional, **keyword):
        keyword.setdefault('float_precision', 'round_trip')
        return original_csv(*positional, **keyword)
    pd.read_csv = read_csv
    try:
        result = summarize(out, cfg)
    finally:
        pd.read_csv = original_csv
    print(json.dumps(dict(status='point_tables_rebuilt', independent_model_fits=0, output=str(out), provenance=result), indent=2))

if __name__ == '__main__':
    main()
