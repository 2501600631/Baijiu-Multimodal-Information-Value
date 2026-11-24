from pathlib import Path
import argparse
import collections
import csv
import gzip
import json
import os
import sys
import time
import traceback
import numpy as np
import pandas as pd
from v8 import runner as base

ROOT=Path(__file__).resolve().parents[1]
MODELS=('sopls_perblock__uv+raman','sopls_perblock__uv+raman+nir')
FIELDS=['stage','condition','repeat','fold','training_size','budget_refit','bootstrap_refit','seed','model','sample_id','target','y_true','y_pred','train_sd','outer_train_sd','n_train','n_unique_train','n_cal','effective_nir_weight','n_acquisition_modalities']

def task_plan(cfg,splits):
    return [dict(stage='v9',condition='sopls_perblock',repeat=int(s['repeat']),fold=int(s['fold']),training_size='full',budget_refit=-1,bootstrap_refit=-1,
                 seed=int(cfg['seed'])+10000*int(s['repeat'])+100*int(s['fold'])) for s in splits]

def task_name(task):
    return 'sopls_perblock_r{repeat}_f{fold}.json.gz'.format(**task)

def design_files():
    names=['configs/sopls_perblock_full.json','docs/V9_SELECTION_RULES.md','docs/V9_STATISTICAL_PLAN.md','docs/V9_SCIENTIFIC_RATIONALE.md']
    return names+[p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'v9').rglob('*.py')) if '__pycache__' not in p.parts]

def frozen_design():
    from repository.verify import verify_design_snapshot
    return verify_design_snapshot(ROOT)

def parent_evidence():
    from repository.verify import parent_evidence as inspect_parent
    return inspect_parent(ROOT)

def execution_context(cfg,dataset):
    design=frozen_design()
    parent=parent_evidence()
    runtime,environment=base.snapshot_environment()
    context=dict(version='9.0.0',config=cfg,frozen_design_files=design['files'],frozen_design_sha256=base.file_hash(ROOT/'provenance/v9_design_freeze.json'),
                 parent_evidence=parent,runtime=runtime,n_bottles=len(dataset.ids),targets=dataset.target_names,
                 fit_scope='Fifteen new per-block structural tasks only; no refitting of the completed V8 tasks')
    return base.digest_object(context),context,environment

def evaluate(dataset,cfg,split,task):
    from v9.engine import evaluate_task
    fit=np.asarray(split['tr'],int)
    test=np.asarray(split['te'],int)
    response=dataset.Y.to_numpy(float)
    train={m:a[fit] for m,a in dataset.X.items()}
    evaluation={m:a[test] for m,a in dataset.X.items()}
    result=evaluate_task(train,response[fit],dataset.ids[fit],evaluation,dataset.axes,cfg,task)
    if set(result['predictions'])!=set(MODELS):
        raise ValueError('The structural task must return the two declared masks')
    sd=response[fit].std(axis=0,ddof=1)
    if np.any(sd<=0): raise ValueError('Every target requires positive fitting variance')
    rows=[]
    for model,prediction in result['predictions'].items():
        prediction=np.asarray(prediction,float)
        if prediction.shape!=(len(test),11) or not np.isfinite(prediction).all():
            raise ValueError('Nonfinite or incomplete structural prediction')
        n_modalities=3 if model.endswith('+nir') else 2
        for i,position in enumerate(test):
            for j,target in enumerate(dataset.target_names):
                rows.append(dict(task,model=model,sample_id=str(dataset.ids[position]),target=target,y_true=float(response[position,j]),y_pred=float(prediction[i,j]),
                                 train_sd=float(sd[j]),outer_train_sd=float(sd[j]),n_train=len(fit),n_unique_train=len(set(dataset.ids[fit])),n_cal=0,
                                 effective_nir_weight=float(n_modalities==3),n_acquisition_modalities=n_modalities))
    return dict(task=task,fit_ids=dataset.ids[fit].tolist(),test_ids=dataset.ids[test].tolist(),targets=dataset.target_names,
                model_names=list(MODELS),diagnostics=result['diagnostics'],predictions=rows)

def validate_payload(payload,dataset,cfg,split,task):
    from executed.legacy_engine import grouped_folds
    fit=np.asarray(split['tr'],int)
    test=np.asarray(split['te'],int)
    fit_ids=dataset.ids[fit].tolist()
    test_ids=dataset.ids[test].tolist()
    if payload['task']!=task or payload['fit_ids']!=fit_ids or payload['test_ids']!=test_ids:
        raise ValueError('Cached task or bottle roles changed')
    if set(fit_ids)&set(test_ids) or len(set(fit_ids))!=len(fit_ids):
        raise ValueError('Invalid or overlapping outer bottle roles')
    if payload['targets']!=dataset.target_names or set(payload['model_names'])!=set(MODELS):
        raise ValueError('Cached target or model inventory changed')
    frame=pd.DataFrame(payload['predictions'])
    if len(frame)!=len(test)*22 or frame.duplicated(['model','sample_id','target']).any():
        raise ValueError('Cached prediction keys are incomplete or duplicated')
    if set(frame.model)!=set(MODELS) or set(frame.target)!=set(dataset.target_names) or set(frame.sample_id)!=set(test_ids):
        raise ValueError('Cached prediction identities differ')
    numeric=['y_true','y_pred','train_sd','outer_train_sd','n_train','n_unique_train','n_cal','effective_nir_weight','n_acquisition_modalities']
    if not np.isfinite(frame[numeric].to_numpy(float)).all(): raise ValueError('Nonfinite cached values')
    truth=[dataset.Y.loc[str(identity),target] for identity,target in zip(frame.sample_id,frame.target)]
    if not np.array_equal(frame.y_true.to_numpy(float),truth): raise ValueError('Cached reference values changed')
    sd=dataset.Y.iloc[fit].std(axis=0,ddof=1)
    for name in ('train_sd','outer_train_sd'):
        if not np.allclose(frame[name],[sd[t] for t in frame.target],atol=1e-12,rtol=0): raise ValueError('Cached response scales changed')
    if not frame.n_train.eq(len(fit)).all() or not frame.n_unique_train.eq(len(fit)).all() or not frame.n_cal.eq(0).all():
        raise ValueError('Cached training or calibration counts changed')
    for name,value in task.items():
        if name not in frame or not frame[name].eq(value).all(): raise ValueError('Cached task metadata changed: '+name)
    diag=payload['diagnostics']
    if diag.get('training_ids')!=fit_ids or int(diag.get('seed',-1))!=task['seed']:
        raise ValueError('Diagnostic training roles changed')
    folds=grouped_folds(np.asarray(fit_ids),int(cfg['inner_folds']),task['seed'])
    if len(diag['inner_splits'])!=len(folds): raise ValueError('Inner fold records are incomplete')
    for recorded,(tr,va) in zip(diag['inner_splits'],folds):
        if recorded['train_ids']!=np.asarray(fit_ids)[tr].tolist() or recorded['validation_ids']!=np.asarray(fit_ids)[va].tolist():
            raise ValueError('Inner fitting or validation roles changed')
        if recorded['train_row_indices']!=tr.tolist() or recorded['validation_row_indices']!=va.tolist():
            raise ValueError('Inner fitting or validation row indices changed')
        expected_sd=dataset.Y.iloc[fit[tr]].std(axis=0,ddof=1).to_numpy(float)
        if not np.allclose(recorded['training_response_sd'],expected_sd,atol=1e-12,rtol=0):
            raise ValueError('Inner fitting response scales changed')
    selections=diag.get('selections',{})
    fits=diag.get('final_fits',{})
    if set(selections)!=set(MODELS) or set(fits)!=set(MODELS): raise ValueError('Final model diagnostics are incomplete')
    for model in MODELS:
        mask=model.split('__',1)[1]
        modalities=mask.split('+')
        group=frame[frame.model.eq(model)]
        if not group.n_acquisition_modalities.eq(len(modalities)).all() or not group.effective_nir_weight.eq(float('nir' in modalities)).all():
            raise ValueError('The fixed-mask acquisition count changed')
        records=selections[model]
        if len(records)!=11 or sorted(x['target_index'] for x in records)!=list(range(11)):
            raise ValueError('All eleven target selections are required')
        final=fits[model]
        if len(final)!=11 or sorted(x['target_index'] for x in final)!=list(range(11)):
            raise ValueError('All eleven final fit records are required')
        for record in records:
            spec=record['spec']
            vector=spec['components_by_block']
            if spec['mask']!=mask or set(spec['order'])!=set(modalities) or len(spec['order'])!=len(modalities):
                raise ValueError('Selected block mask or order changed')
            if len(vector)!=len(modalities) or any(type(k) is not int or k not in cfg['perblock']['components'] for k in vector):
                raise ValueError('Selected per-block components are outside the frozen grid')
            if not np.isfinite(record['inner_mse']): raise ValueError('Selected inner score is invalid')
            for modality in modalities:
                if spec['preprocessing'][modality] not in cfg['sopls_preprocessing_options'][modality]:
                    raise ValueError('Selected preprocessing is outside the frozen options')
            if spec.get('components')!=vector or spec.get('component_map')!=dict(zip(spec['order'],vector)):
                raise ValueError('Component vector aliases or modality mapping changed')
            matching=[item for item in final if item['target_index']==record['target_index']][0]
            if matching['spec']!=spec:
                raise ValueError('Selected and fitted specifications disagree')
            details=matching['details']
            if details.get('block_order')!=spec['order']:
                raise ValueError('Final fitted block order disagrees with the selection')
            blocks=details.get('blocks',[])
            if [item['modality'] for item in blocks]!=spec['order'] or [item['requested_components'] for item in blocks]!=vector:
                raise ValueError('Final fitted component requests disagree with the selection')
            stages=record.get('stage_inner_scores',[])
            if len(stages)!=len(vector) or any(item['stage']!=j+1 or item['modality']!=spec['order'][j] or item['selected_k']!=vector[j] for j,item in enumerate(stages)):
                raise ValueError('Greedy stage records disagree with the selected vector')
            if stages[-1]['inner_mse']!=record['inner_mse']:
                raise ValueError('Final greedy stage score disagrees with the selected score')
        for record in final:
            if record['fit_ids']!=fit_ids: raise ValueError('Final fit identities changed')
    return True

def checked_cache(path,signature,dataset,cfg,split,task):
    wrapper=base.read_json(path)
    if wrapper.get('signature')!=signature: raise ValueError('Extension cache signature mismatch')
    if wrapper.get('payload_sha256')!=base.digest_object(wrapper['payload']): raise ValueError('Extension payload hash mismatch')
    validate_payload(wrapper['payload'],dataset,cfg,split,task)
    return wrapper['payload']

def consolidate(out,plan,signature,dataset,cfg,lookup,parent_status,failure=None):
    finished=[]
    count=0
    temporary=out/'predictions.csv.gz.tmp'
    with gzip.open(str(temporary),'wt',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=FIELDS)
        writer.writeheader()
        for task in plan:
            path=out/'tasks'/task_name(task)
            if not path.exists(): continue
            try:
                payload=checked_cache(path,signature,dataset,cfg,lookup[(task['repeat'],task['fold'])],task)
            except Exception as exc:
                if failure is None: raise
                failure.setdefault('invalid_cached_tasks',[]).append(dict(task=task,error=type(exc).__name__+': '+str(exc)))
                continue
            writer.writerows(payload['predictions'])
            finished.append(task)
            count+=len(payload['predictions'])
    os.replace(str(temporary),str(out/'predictions.csv.gz'))
    stages=dict(parent_status['stages'])
    stages['sopls_perblock']=dict(expected_tasks=15,completed_tasks=len(finished),complete=len(finished)==15)
    status='failed' if failure else ('all_requested_stages_complete' if len(finished)==15 else 'incomplete')
    result=dict(status=status,signature=signature,expected_tasks=270,completed_tasks=255+len(finished),
                parent_completed_tasks=255,extension_expected_tasks=15,extension_completed_tasks=len(finished),
                extension_status='failed' if failure else ('complete' if len(finished)==15 else 'incomplete'),
                prediction_rows=count,parent_prediction_rows=int(parent_status['prediction_rows']),
                combined_v8_and_extension_prediction_rows=int(parent_status['prediction_rows'])+count,stages=stages,
                all_eleven_targets_retained=True,independent_bottles=len(dataset.ids),external_validation=False)
    if failure: result['failure']=failure
    base.atomic_json(out/'COMPLETENESS.json',result)
    return result

def main(argv=None):
    parser=argparse.ArgumentParser(description='Run the frozen fifteen-task per-block SO-PLS structural extension')
    parser.add_argument('--stage',choices=('all','prepare','summarize'),default='all')
    parser.add_argument('--out')
    parser.add_argument('--max-tasks',type=int)
    args=parser.parse_args(argv)
    cfg=base.read_json(ROOT/'configs/sopls_perblock_full.json')
    if cfg['perblock']['components']!=[1,2,3,5] or cfg['inner_folds']!=3 or cfg['repeats']!=3 or cfg['outer_folds']!=5:
        raise ValueError('The fixed structural experiment has changed')
    if args.max_tasks is not None and args.max_tasks<0: raise ValueError('The task limit cannot be negative')
    out=Path(args.out).resolve() if args.out else ROOT/'runs/reproduce_structural'
    protected=[ROOT/name for name in ('executed','v8','v9','results','data','figures','provenance','configs','docs','tools','results/robustness')]
    if out==ROOT or any(out==p or p in out.parents for p in protected):
        raise ValueError('Historical sources and results are immutable; choose a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    with base.output_lock(out):
        dataset=base.load_dataset(ROOT/'executed/data')
        if len(dataset.ids)!=139 or len(dataset.target_names)!=11: raise ValueError('All 139 bottles and eleven targets are required')
        parent_protocol=base.read_json(ROOT/'results/robustness/protocol.json')
        splits=base.historical_splits(dataset.ids,parent_protocol['context']['config'])
        plan=task_plan(cfg,splits)
        if len(plan)!=15 or len({task_name(t) for t in plan})!=15: raise ValueError('Exactly fifteen unique tasks are required')
        signature,context,environment=execution_context(cfg,dataset)
        previous=out/'protocol.json'
        if previous.exists() and base.read_json(previous).get('signature')!=signature:
            raise ValueError('Source, frozen design, parent evidence or scientific runtime changed; existing runs are preserved')
        base.atomic_json(previous,dict(signature=signature,context=context))
        base.atomic_json(out/'environment.json',environment)
        if environment.get('pip_freeze') is not None: (out/'environment.freeze.txt').write_text(environment['pip_freeze'],encoding='utf-8')
        base.atomic_json(out/'task_plan.json',plan)
        base.atomic_json(out/'splits.json',[{k:v for k,v in split.items() if k not in ('tr','te')} for split in splits])
        (out/'tasks').mkdir(exist_ok=True)
        lookup={(s['repeat'],s['fold']):s for s in splits}
        parent_status=context['parent_evidence']['parent_status']
        executed=0
        if args.stage=='all':
            for index,task in enumerate(plan,1):
                path=out/'tasks'/task_name(task)
                try:
                    if path.exists():
                        checked_cache(path,signature,dataset,cfg,lookup[(task['repeat'],task['fold'])],task)
                        print('[%d/15] verified cached %s'%(index,path.name),flush=True)
                        continue
                    if args.max_tasks is not None and executed>=args.max_tasks: break
                    print('[%d/15] running %s'%(index,path.name),flush=True)
                    started=time.time()
                    payload=evaluate(dataset,cfg,lookup[(task['repeat'],task['fold'])],task)
                    validate_payload(payload,dataset,cfg,lookup[(task['repeat'],task['fold'])],task)
                    base.atomic_json(path,dict(signature=signature,payload_sha256=base.digest_object(payload),payload=payload))
                    executed+=1
                    print('completed %.1f seconds; eleven targets and both masks saved'%(time.time()-started),flush=True)
                except Exception as exc:
                    failure=dict(task=task,error_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),policy='Run stopped; no candidate or outer-label fallback')
                    base.atomic_json(out/'failure.json',failure)
                    consolidate(out,plan,signature,dataset,cfg,lookup,parent_status,failure=failure)
                    raise
        report=consolidate(out,plan,signature,dataset,cfg,lookup,parent_status)
        if args.stage in ('all','summarize') and report['extension_completed_tasks']:
            from v9.analysis import analyze
            analyze(ROOT,out,cfg)
        print(json.dumps(report,indent=2),flush=True)
        return report

if __name__=='__main__':
    main()
