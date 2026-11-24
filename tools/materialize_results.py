from pathlib import Path
import argparse
import gzip
import shutil

ROOT = Path(__file__).resolve().parents[1]

def materialize(destination):
    source = ROOT / 'results' / 'baseline'
    destination = Path(destination).resolve()
    if destination == source.resolve() or source.resolve() in destination.parents:
        raise ValueError('Do not expand over the recorded archive')
    for path in source.rglob('*'):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if path.name.endswith('.csv.gz'):
            target = destination / relative.with_suffix('')
            target.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(str(path), 'rb') as incoming, target.open('wb') as output:
                shutil.copyfileobj(incoming, output)
        else:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(path), str(target))
    print(str(destination))
    return destination

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Expand recorded point predictions into an editable local working copy')
    parser.add_argument('--out', default=str(ROOT / 'work' / 'expanded_results'))
    materialize(parser.parse_args().out)
