from pathlib import Path
import gzip
import json
import shutil
from data_io import sha256
from legacy_summary import summarize

ROOT = Path(__file__).resolve().parent

def summarize_archive():
    archive = ROOT / 'results' / 'archived_v4'
    out = ROOT / 'results' / 'legacy_view'
    record = json.loads((archive / 'legacy_execution.json').read_text(encoding='utf-8'))
    storage = json.loads((archive / 'archive_storage.json').read_text(encoding='utf-8'))
    expected = record['archived_result_sha256']
    if set(storage['files']) != set(expected):
        raise ValueError('Compressed archive does not contain exactly the recorded historical result files')
    for name, digest in expected.items():
        item = storage['files'][name]
        if item['gzip_path'] != name + '.gz' or item['uncompressed_sha256'] != digest:
            raise ValueError('Historical archive mapping differs: ' + name)
        source = archive / item['gzip_path']
        if sha256(source) != item['gzip_sha256']:
            raise ValueError('Compressed archive checksum differs: ' + name)
        destination = out / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + '.tmp')
        with gzip.open(str(source), 'rb') as handle, temporary.open('wb') as target:
            shutil.copyfileobj(handle, target)
        if sha256(temporary) != digest:
            temporary.unlink()
            raise ValueError('Decoded historical result checksum differs: ' + name)
        temporary.replace(destination)
    shutil.copy2(archive / 'COMPLETENESS.json', out / 'COMPLETENESS.json')
    selection = archive / 'derived' / 'selection_frequency.csv'
    if selection.exists():
        (out / 'derived').mkdir(exist_ok=True)
        shutil.copy2(selection, out / 'derived' / selection.name)
    cfg = dict(record['config'])
    cfg['summary_make_figures'] = False
    cfg['archived_selection_frequency'] = True
    result = summarize(out, cfg)
    result['historical_execution_signature'] = record['signature']
    result['archive_uncompressed_sha256'] = expected
    result['archive_is_not_a_new_fit'] = True
    (out / 'derived' / 'archive_view_provenance.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Verified and summarized historical saved predictions without fitting models.')
    print('English reports: results/legacy_view/derived/all_11_targets_report.md')
    return result

if __name__ == '__main__':
    summarize_archive()
