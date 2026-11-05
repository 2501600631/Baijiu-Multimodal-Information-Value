import sys, importlib, json
from pathlib import Path
from data_io import sha256, load_dataset
print('Python:', sys.version)
if not (3, 8) <= sys.version_info[:2] <= (3, 11):
    raise RuntimeError('Use Python3.8-3.11 with this pinned package; Python3.10 or3.11 is recommended for a new environment.')
for name in ['numpy', 'scipy', 'pandas', 'sklearn', 'matplotlib', 'openpyxl']:
    m = importlib.import_module(name)
    print(name, getattr(m, '__version__', 'unknown'))
p = Path(__file__).resolve().parent
for name, digest in json.loads((p / 'provenance' / 'inputs.json').read_text(encoding='utf-8'))['converted_input_sha256'].items():
    assert sha256(p / 'data' / name) == digest, name
data = load_dataset(p / 'data')
assert data.Y.shape == (139, 11)
print('Dependencies, input hashes and 139-bottle alignment passed. No model fitted.')
