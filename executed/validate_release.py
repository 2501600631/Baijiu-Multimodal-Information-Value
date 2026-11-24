from pathlib import Path
import ast
import gzip
import hashlib
import io
import json
import re
import tokenize
from data_io import load_dataset, sha256

ROOT=Path(__file__).resolve().parent
EXCLUDED={'__pycache__','.git','.venv','.pytest_cache','legacy_view'}
TEXT_EXT={'.py','.md','.txt','.json','.csv','.yaml','.yml','.cff','.bat','.gitignore','.log'}


def included(path):
    return not any(p in EXCLUDED or p.startswith('_synthetic_') or p.startswith('new_') for p in path.relative_to(ROOT).parts)


def release_files():
    return sorted(p for p in ROOT.rglob('*') if p.is_file() and included(p) and p.name!='PACKAGE_MANIFEST.json')


def validate():
    required=['run.py','engine.py','legacy_engine.py','preprocessing.py','sopls.py','uncertainty.py','summarize.py','plots.py','README.md','run_full.bat','configs/full.json','configs/quick.json','docs/methods_and_decisions.md','docs/literature_basis.md','tests/test_runner.py']
    for name in required:
        if not (ROOT/name).is_file():
            raise ValueError('Required release file is absent: '+name)
    files=release_files()
    checked=0
    for path in files:
        name=path.relative_to(ROOT).as_posix()
        if re.search('[\u3400-\u9fff]',name):
            raise ValueError('Non-English filename: '+name)
        if path.suffix.lower() not in TEXT_EXT and path.name!='.gitignore':
            continue
        text=path.read_text(encoding='utf-8-sig')
        if re.search('[\u3400-\u9fff]',text):
            raise ValueError('Non-English text: '+name)
        if re.search(r'(?:sk-[A-Za-z0-9]{25,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)',text):
            raise ValueError('Potential secret: '+name)
        if re.search(r'[A-Za-z]:[/\\]Users[/\\][^/\\\s]+',text):
            raise ValueError('Private absolute source path: '+name)
        if path.suffix=='.py':
            if any(t.type==tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(text).readline)):
                raise ValueError('Python comment remains: '+name)
            tree=ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node,(ast.Module,ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)) and ast.get_docstring(node) is not None:
                    raise ValueError('Python docstring remains: '+name)
            compile(text,str(path),'exec')
        checked+=1
    inputs=json.loads((ROOT/'provenance'/'inputs.json').read_text(encoding='utf-8'))
    for name,digest in inputs['converted_input_sha256'].items():
        if sha256(ROOT/'data'/name)!=digest:
            raise ValueError('Lossless source measurement hash differs: '+name)
    dataset=load_dataset(ROOT/'data')
    if dataset.Y.shape!=(139,11):
        raise ValueError('The139-independent-bottle eleven-target panel is incomplete')
    archive=ROOT/'results'/'archived_v4'
    legacy=json.loads((archive/'legacy_execution.json').read_text(encoding='utf-8'))
    storage=json.loads((archive/'archive_storage.json').read_text(encoding='utf-8'))
    if set(storage['files'])!=set(legacy['archived_result_sha256']):
        raise ValueError('Historical result archive membership changed')
    for name,digest in legacy['archived_result_sha256'].items():
        record=storage['files'][name]
        path=archive/record['gzip_path']
        if record['uncompressed_sha256']!=digest or sha256(path)!=record['gzip_sha256']:
            raise ValueError('Historical compressed archive differs: '+name)
        decoded=hashlib.sha256()
        with gzip.open(str(path),'rb') as handle:
            for block in iter(lambda:handle.read(1048576),b''):
                decoded.update(block)
        if decoded.hexdigest()!=digest:
            raise ValueError('Historical decoded stream differs: '+name)
    manifest=ROOT/'PACKAGE_MANIFEST.json'
    if manifest.exists():
        saved=json.loads(manifest.read_text(encoding='utf-8'))['sha256']
        if set(saved)!=set(p.relative_to(ROOT).as_posix() for p in files):
            raise ValueError('Release membership changed; use a separate output directory')
        for name,digest in saved.items():
            if sha256(ROOT/name)!=digest:
                raise ValueError('Release file changed: '+name)
    return dict(english_text_files=checked,release_files=len(files),input_bottles=139,targets=11,original_input_hashes_verified=len(inputs['converted_input_sha256']),historical_raw_streams_verified=4,comments_removed=True,docstrings_removed=True,no_models_fitted_by_validation=True)


if __name__=='__main__':
    print(json.dumps(validate(),indent=2))
