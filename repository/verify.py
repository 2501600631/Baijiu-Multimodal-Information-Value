from pathlib import Path
import argparse
import gzip
import hashlib
import json
import sys

SECTIONS=('source_files','configuration_files','data_files','result_files')
RUN_NAMES=('baseline','robustness','structural')

def sha256(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):
            value.update(block)
    return value.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('utf-8')).hexdigest()

def read_json(path):
    path=Path(path)
    with (gzip.open(str(path),'rt',encoding='utf-8') if path.suffix=='.gz' else path.open(encoding='utf-8')) as stream:
        return json.load(stream)

def inside(root,name):
    root=Path(root).resolve()
    destination=(root/name).resolve()
    if destination==root or root not in destination.parents:
        raise ValueError('Manifest paths must stay inside the repository')
    return destination

def verify_snapshot(root,sections=None):
    root=Path(root).resolve()
    manifest=read_json(root/'provenance/public_representation.json')
    context=manifest['context']
    if manifest['representation_signature']!=digest(context):
        raise ValueError('Representation manifest signature mismatch')
    sections=SECTIONS if sections is None else tuple(sections)
    if not set(sections).issubset(SECTIONS):
        raise ValueError('Unknown representation manifest section')
    checked={}
    for section in sections:
        records=context[section]
        if not records:
            raise ValueError('Representation section is empty: '+section)
        for name,expected in records.items():
            path=inside(root,name)
            if not path.is_file() or sha256(path)!=expected:
                raise ValueError('Representation file mismatch: '+name)
        checked[section]=len(records)
    return dict(status='pass',manifest=manifest,checked_sections=checked,
                interpretation='Current representation integrity; recorded execution signatures are retained separately')

def verify_design_snapshot(root):
    root=Path(root).resolve()
    snapshot=read_json(root/'provenance/public_design_snapshot.json')
    if len(snapshot.get('original_design_sha256',''))!=64:
        raise ValueError('The original design digest is missing')
    for group in ('files','parent_files'):
        if not snapshot[group]:
            raise ValueError('The current design snapshot is incomplete')
        for name,expected in snapshot[group].items():
            path=inside(root,name)
            if not path.is_file() or sha256(path)!=expected:
                raise ValueError('Design representation mismatch: '+name)
    return snapshot

def verify_checksums(root):
    root=Path(root).resolve()
    checked=[]
    for line in (root/'CHECKSUMS.sha256').read_text(encoding='utf-8').splitlines():
        if not line:
            continue
        expected,name=line.split('  ',1)
        if len(expected)!=64 or name in checked:
            raise ValueError('Invalid or duplicated distribution checksum entry')
        path=inside(root,name)
        if not path.is_file() or sha256(path)!=expected:
            raise ValueError('Distribution checksum mismatch: '+name)
        checked.append(name)
    if not checked:
        raise ValueError('The distribution checksum inventory is empty')
    return len(checked)

def verify_imported_run(root,name,deep=False):
    root=Path(root).resolve()
    if name not in RUN_NAMES:
        raise ValueError('Unknown imported run: '+name)
    manifest=read_json(root/'provenance/public_representation.json')
    if digest(manifest['context'])!=manifest['representation_signature']:
        raise ValueError('Representation manifest signature mismatch')
    item=manifest['context']['imported_runs'][name]
    protocol_path=inside(root,item['protocol_path'])
    if sha256(protocol_path)!=item['public_protocol_sha256']:
        raise ValueError('Imported protocol representation changed: '+name)
    protocol=read_json(protocol_path)
    if digest(protocol['context'])!=item['public_context_sha256']:
        raise ValueError('Imported context representation changed: '+name)
    signature=item['original_execution_signature']
    if protocol.get('signature')!=signature or len(item['source_protocol_sha256'])!=64:
        raise ValueError('Recorded execution identity changed: '+name)
    folder=inside(root,item['tasks_path'])
    paths=sorted(folder.glob('*.json.gz'))
    expected=int(item['task_count'])
    if len(paths)!=expected:
        raise ValueError('Imported task count differs from its declared scope: '+name)
    task_count=0
    prediction_rows=0
    if deep:
        sys.path.insert(0,str(root/'executed'))
        import run as baseline
        from data_io import load_dataset
        dataset=load_dataset(root/'executed/data')
        cfg=protocol['context']['config']
        splits=baseline.make_splits(dataset.ids,cfg)
        lookup={(split['repeat'],split['fold']):split for split in splits}
        plan=read_json(folder.parent/'task_plan.json')
        if len(plan)!=expected:
            raise ValueError('Imported task-plan count differs: '+name)
        if name=='robustness':
            from v8 import runner as implementation
        elif name=='structural':
            from v9 import runner as implementation
        else:
            implementation=baseline
        names=[]
        for task in plan:
            filename=implementation.task_name(task)
            if filename in names:
                raise ValueError('Duplicated imported task identity: '+name)
            names.append(filename)
            path=folder/filename
            wrapper=read_json(path)
            if wrapper.get('signature')!=signature:
                raise ValueError('Recorded task execution signature differs: '+name+'/'+filename)
            if baseline.payload_hash(wrapper['payload'])!=wrapper.get('payload_sha256'):
                raise ValueError('Recorded task payload digest differs: '+name+'/'+filename)
            split=lookup[(task['repeat'],task['fold'])]
            if name=='baseline':
                payload=baseline.read_cache(path,signature,task)
                baseline.validate_payload(payload,dataset,cfg,split,task)
            else:
                payload=implementation.checked_cache(path,signature,dataset,cfg,split,task)
            prediction_rows+=len(payload['predictions'])
            task_count+=1
        if set(names)!={path.name for path in paths}:
            raise ValueError('Imported file and task-plan identities differ: '+name)
        if prediction_rows!=int(item['prediction_rows']):
            raise ValueError('Imported prediction-row count differs: '+name)
    return dict(name=name,status='pass',task_count=expected,payloads_semantically_checked=task_count,
                expected_prediction_rows=int(item['prediction_rows']),prediction_rows_semantically_checked=prediction_rows,
                original_execution_signature=signature,
                current_protocol_representation_sha256=item['public_protocol_sha256'],
                source_protocol_sha256=item['source_protocol_sha256'],
                record_scope='Imported completed numerical execution; current source representation has a separate attestation')

def parent_evidence(root):
    root=Path(root).resolve()
    snapshot=verify_snapshot(root)
    imported=verify_imported_run(root,'robustness',deep=False)
    records=snapshot['manifest']['context']['imported_runs']
    item=records['robustness']
    folder=inside(root,item['tasks_path']).parent
    status=read_json(folder/'COMPLETENESS.json')
    if status['status']!='all_requested_stages_complete' or int(status['completed_tasks'])!=255:
        raise ValueError('All 255 preserved robustness tasks are required')
    if status['signature']!=imported['original_execution_signature']:
        raise ValueError('The preserved parent completion identity changed')
    registry=read_json(folder/'analysis_registry.json')
    prediction=folder/registry['source']
    if sha256(prediction)!=registry['source_sha256']:
        raise ValueError('The preserved parent prediction digest changed')
    for item in registry['tables'].values():
        if sha256(folder/item['path'])!=item['sha256']:
            raise ValueError('A preserved parent table digest changed')
    task_paths=sorted((folder/'tasks').glob('*.json.gz'))
    return dict(parent_signature=imported['original_execution_signature'],parent_completed_tasks=255,
                parent_prediction_rows=int(status['prediction_rows']),parent_prediction_sha256=sha256(prediction),
                parent_task_files={path.name:sha256(path) for path in task_paths},
                historical_prediction_sha256=sha256(root/'results/baseline/benchmark/predictions.csv.gz'),parent_status=status,
                parent_representation_signature=snapshot['manifest']['representation_signature'],
                parent_attestation_scope='Imported numerical execution plus separately checked current representation')

def main(argv=None,root=None):
    parser=argparse.ArgumentParser(description='Verify source, data, completed results and their format-migration records without fitting models')
    parser.add_argument('--deep',action='store_true')
    parser.add_argument('--run',choices=('all',)+RUN_NAMES,default='all')
    args=parser.parse_args(argv)
    root=Path(root).resolve() if root is not None else Path(__file__).resolve().parents[1]
    files=verify_checksums(root)
    snapshot=verify_snapshot(root)
    design=verify_design_snapshot(root)
    selected=RUN_NAMES if args.run=='all' else (args.run,)
    runs=[verify_imported_run(root,name,deep=args.deep) for name in selected]
    report=dict(status='pass',distribution_files_checked=files,current_representation_signature=snapshot['manifest']['representation_signature'],
                representation_sections=snapshot['checked_sections'],design_source_files_checked=len(design['files']),
                imported_runs=runs,deep=args.deep,new_model_fits_performed=0,
                scope='Current representation verified separately from recorded numerical executions')
    print(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    main()
