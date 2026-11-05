from pathlib import Path
import argparse
import json
import shutil
from repository.verify import read_json,verify_snapshot

ROOT=Path(__file__).resolve().parents[1]

def stage_copy(source,destination,with_tasks=False):
    destination.mkdir(parents=True,exist_ok=True)
    for name in ('predictions.csv.gz','COMPLETENESS.json','splits.json'):
        if (source/name).exists(): shutil.copy2(source/name,destination/name)
    if with_tasks: shutil.copytree(source/'tasks',destination/'tasks',dirs_exist_ok=True)

def main(argv=None):
    parser=argparse.ArgumentParser(description='Rebuild statistical summaries from completed predictions without fitting models')
    parser.add_argument('--out',type=Path,default=ROOT/'work/rebuilt_reports')
    args=parser.parse_args(argv)
    out=args.out.resolve()
    for name in ('results','data','executed','v8','v9','repository','figures','configs','provenance','docs','tools'):
        protected=ROOT/name
        if out==protected or protected in out.parents: raise ValueError('Retained evidence is immutable; choose a separate work directory')
    if out==ROOT: raise ValueError('The repository root cannot be used as an output directory')
    verify_snapshot(ROOT)
    from v8.analysis import analyze_archive,analyze_run
    from v9.analysis import analyze
    cfg8=read_json(ROOT/'configs/v8_full.json')
    cfg9=read_json(ROOT/'configs/sopls_perblock_full.json')
    baseline=analyze_archive(ROOT,out/'baseline_audit',cfg8)
    stage_copy(ROOT/'results/robustness',out/'robustness')
    robustness=analyze_run(ROOT,out/'robustness',cfg8)
    stage_copy(ROOT/'results/structural',out/'structural',with_tasks=True)
    structural=analyze(ROOT,out/'structural',cfg9)
    record=dict(status='pass',output=str(out),baseline_tables=len(baseline['tables']),robustness_tables=len(robustness['tables']),structural_tables=len(structural['tables']),model_fits_performed=0)
    (out/'REBUILD_STATUS.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,indent=2))
    return record

if __name__=='__main__':
    main()
