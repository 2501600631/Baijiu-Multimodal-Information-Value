from pathlib import Path
import argparse
import importlib
import importlib.metadata
import json
import platform
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description='Check the supported Python runtime and scientific dependencies')
    parser.add_argument('--python-only',action='store_true')
    args=parser.parse_args()
    if not (3,8)<=sys.version_info[:2]<=(3,11):
        print('Use Python 3.8, 3.9, 3.10 or 3.11 with the pinned scientific dependencies. Python 3.12 and newer are not supported by these pins.')
        return 1
    if args.python_only:
        print('Supported Python: '+platform.python_version())
        return 0
    names={'numpy':'numpy','scipy':'scipy','pandas':'pandas','scikit-learn':'sklearn','matplotlib':'matplotlib','openpyxl':'openpyxl','Pillow':'PIL','threadpoolctl':'threadpoolctl'}
    expected={line.split('==',1)[0]:line.split('==',1)[1] for line in (ROOT/'requirements.txt').read_text().splitlines() if '==' in line}
    versions={}
    failures=[]
    for distribution,module in names.items():
        try:
            importlib.import_module(module)
            versions[distribution]=importlib.metadata.version(distribution)
        except Exception as exc:
            failures.append(distribution+': '+type(exc).__name__+': '+str(exc))
    mismatches={name:dict(installed=version,pinned=expected.get(name)) for name,version in versions.items() if version!=expected.get(name)}
    report=dict(status='fail' if failures else 'pass',python=platform.python_version(),executable=sys.executable,versions=versions,import_failures=failures,pin_mismatches=mismatches)
    print(json.dumps(report,indent=2))
    if failures:
        print('Run 01_install_isolated.bat to install the dependencies in the local .venv folder.')
        return 1
    if mismatches:
        print('Dependencies import successfully, but versions differ from the pins. Use 01_install_isolated.bat for the declared environment. Each experiment records its actual runtime.')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
