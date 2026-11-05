from pathlib import Path
import argparse
import contextlib
import csv
import gzip
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
import numpy as np
import pandas as pd
import scipy
import threadpoolctl

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'executed'))
from data_io import load_dataset
from run import make_splits as historical_splits

CONDITIONS=('unscaled','nir_extended','nir_intervals','balanced_budget','nir_weight','residual','nir_weight_learning','nir_weight_stability')
PAIR_MODELS=[a+'__'+m for a in ('pls','ridge','krr','sopls') for m in ('uv+raman','uv+raman+nir')]

def clean(value):
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    if isinstance(value,np.ndarray): return clean(value.tolist())
    if isinstance(value,(np.integer,)): return int(value)
    if isinstance(value,(np.bool_,)): return bool(value)
    if isinstance(value,(float,np.floating)):
        if not np.isfinite(value): raise ValueError('Nonfinite values cannot enter a completed task')
        return float(value)
    if isinstance(value,Path): return str(value)
    return value

def digest_object(value):
    return hashlib.sha256(json.dumps(clean(value),sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('utf-8')).hexdigest()

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def atomic_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(str(temporary),'wt',encoding='utf-8') as f:
        json.dump(clean(value),f,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
    os.replace(str(temporary),str(path))

def read_json(path):
    path=Path(path)
    with (gzip.open(str(path),'rt',encoding='utf-8') if path.suffix=='.gz' else path.open(encoding='utf-8')) as f:
        return json.load(f)

def snapshot_environment():
    packages={}
    for name in ('numpy','scipy','pandas','scikit-learn','matplotlib','openpyxl','Pillow','threadpoolctl'):
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name]=None
    stream=io.StringIO()
    with contextlib.redirect_stdout(stream): np.show_config()
    pools=threadpoolctl.threadpool_info()
    stable_pools=[{k:p.get(k) for k in ('user_api','internal_api','prefix','version','num_threads','threading_layer','architecture')} for p in pools]
    stable=dict(python=platform.python_version(),packages=packages,system=platform.system(),machine=platform.machine(),
                numpy_configuration=stream.getvalue(),threadpools=stable_pools,
                requested_threads={name:os.environ.get(name) for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},
                float64=dict(bits=np.finfo(np.float64).bits,eps=np.finfo(np.float64).eps,max=np.finfo(np.float64).max),
                float_info=dict(mant_dig=sys.float_info.mant_dig,rounds=sys.float_info.rounds),numpy_error_policy=np.geterr())
    detail=dict(stable,executable=sys.executable,platform=platform.platform(),processor=platform.processor(),threadpool_details=pools,
                captured_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),historical_runtime=False)
    try:
        process=subprocess.run([sys.executable,'-m','pip','freeze','--all'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30)
        detail['pip_freeze']=process.stdout
        detail['pip_freeze_exit_code']=process.returncode
        detail['pip_freeze_error']=process.stderr
    except (OSError,subprocess.TimeoutExpired) as exc:
        detail['pip_freeze']=None
        detail['pip_freeze_error']=type(exc).__name__+': '+str(exc)
    return stable,detail

def execution_context(cfg,dataset):
    source={str(p.relative_to(ROOT)).replace('\\','/'):file_hash(p) for folder in ('v8','executed') for p in sorted((ROOT/folder).rglob('*.py')) if '__pycache__' not in p.parts}
    inputs={str(p.relative_to(ROOT)).replace('\\','/'):file_hash(p) for p in sorted((ROOT/'executed/data').glob('*')) if p.is_file()}
    stable,environment=snapshot_environment()
    archive_inputs={p.name:file_hash(p) for p in sorted((ROOT/'results/baseline/tasks').glob('benchmark_*.json.gz'))}
    if len(archive_inputs)!=15: raise ValueError('All fifteen historical benchmark payloads are required')
    context=dict(version='8.0.0',config=cfg,source_files=source,input_files=inputs,archived_benchmark_files=archive_inputs,runtime=stable,
                 n_bottles=len(dataset.ids),targets=dataset.target_names,
                 historical_protocol_sha256=file_hash(ROOT/'provenance/executed_protocol.json'),
                 analysis_scope='Retrospective same-cohort extension with frozen archived bottle identities')
    return digest_object(context),context,environment

def plan_tasks(cfg,splits):
    tasks=[]
    extension=cfg['v8']
    for split in splits:
        r,f=int(split['repeat']),int(split['fold'])
        base=dict(stage='v8',repeat=r,fold=f,training_size='full',budget_refit=-1,bootstrap_refit=-1,
                  seed=int(cfg['seed'])+10000*r+100*f)
        for condition in ('unscaled','nir_extended','nir_intervals','nir_weight','residual'):
            tasks.append(dict(base,condition=condition))
        for draw in range(int(extension['balanced_draws'])):
            tasks.append(dict(base,condition='balanced_budget',budget_refit=draw))
        for size in cfg['learning_sizes']:
            if int(size)!=size or size<max(6,2*int(cfg['inner_folds'])) or size>=len(split['tr']):
                raise ValueError('Invalid reduced training budget')
            tasks.append(dict(base,condition='nir_weight_learning',training_size=str(int(size))))
        for draw in range(int(extension['weight_bootstrap_refits'])):
            tasks.append(dict(base,condition='nir_weight_stability',budget_refit=draw,bootstrap_refit=draw))
    return sorted(tasks,key=lambda t:(CONDITIONS.index(t['condition']),t['repeat'],t['fold'],t['training_size'],t['budget_refit']))

def task_name(task):
    return '{condition}_r{repeat}_f{fold}_n{training_size}_b{budget_refit}.json.gz'.format(**task)

def roles(dataset,split,task):
    fit=np.asarray(split['tr'],dtype=int).copy()
    if task['condition']=='nir_weight_learning':
        fit=np.random.RandomState(task['seed']+700000).permutation(fit)[:int(task['training_size'])]
    if task['condition']=='nir_weight_stability':
        fit=np.random.RandomState(task['seed']+100000+task['budget_refit']).choice(fit,len(fit),replace=True)
    return fit,np.asarray(split['te'],dtype=int)

def expected_models(condition,cfg):
    if condition.startswith('nir_weight'): return ['krr__weight_zero','krr__weight_selected']
    if condition=='residual': return ['ridge_residual__zero','ridge_residual__selected']
    return [a+'__'+m for a in cfg['algorithms'] for m in ('uv+raman','uv+raman+nir')]

def evaluate(dataset,cfg,split,task):
    from v8.engine import evaluate_task
    from v8.residual import evaluate_residual
    fit,test=roles(dataset,split,task)
    train_x={m:a[fit] for m,a in dataset.X.items()}
    eval_x={m:a[test] for m,a in dataset.X.items()}
    Y=dataset.Y.to_numpy(float)
    original=None
    if task['condition']=='balanced_budget' and cfg['profile']=='full':
        source=ROOT/'results/baseline/tasks'/('benchmark_r%d_f%d_cenhanced_nfull_b-1.json.gz'%(task['repeat'],task['fold']))
        original=read_json(source)
        if original['payload_sha256']!=digest_object(original['payload']):
            from run import payload_hash
            if original['payload_sha256']!=payload_hash(original['payload']): raise ValueError('Archived payload hash mismatch')
    local=dict(task)
    if task['condition'].startswith('nir_weight'): local['condition']='nir_weight'
    if task['condition']=='residual':
        result=evaluate_residual(train_x,Y[fit],dataset.ids[fit],eval_x,dataset.axes,cfg,local)
    else:
        result=evaluate_task(train_x,Y[fit],dataset.ids[fit],eval_x,dataset.axes,cfg,local,archived_payload=original)
    predictions=result['predictions']
    if set(predictions)!=set(expected_models(task['condition'],cfg)): raise ValueError('Unexpected completed model set')
    sd=Y[fit].std(axis=0,ddof=1)
    outer_sd=Y[split['tr']].std(axis=0,ddof=1)
    if np.any(sd<=0) or np.any(outer_sd<=0): raise ValueError('Nonpositive target scale')
    acquisition=result['diagnostics'].get('acquisition_by_model',{})
    rows=[]
    for model,prediction in predictions.items():
        prediction=np.asarray(prediction,float)
        if prediction.shape!=(len(test),11) or not np.isfinite(prediction).all(): raise ValueError('Invalid prediction array')
        item=acquisition.get(model,{})
        if model.endswith('__uv+raman') or model.endswith('__weight_zero') or model.endswith('__zero'):
            inputs=['uv','raman']
        else: inputs=item.get('modality_union',['uv','raman','nir'])
        n_inputs=item.get('n_modalities',len(inputs))
        weight=item.get('effective_nir_weight',item.get('nir_weight'))
        if weight is None:
            records=result['diagnostics'].get('selections',{}).get(model,[])
            if records:
                spec=records[0].get('spec',{})
                weight=spec.get('nir_weight',spec.get('multiplier',spec.get('correction_multiplier')))
        if weight is None: weight=0.0 if 'nir' not in inputs else 1.0
        if float(weight)==0 and int(n_inputs)!=2: raise ValueError('Zero NIR contribution must require only two modalities')
        for i,position in enumerate(test):
            for j,target in enumerate(dataset.target_names):
                rows.append(dict(task,model=model,sample_id=str(dataset.ids[position]),target=target,y_true=float(Y[position,j]),
                                 y_pred=float(prediction[i,j]),train_sd=float(sd[j]),outer_train_sd=float(outer_sd[j]),
                                 n_train=len(fit),n_unique_train=len(np.unique(dataset.ids[fit])),n_cal=0,
                                 effective_nir_weight=float(weight),n_acquisition_modalities=int(n_inputs)))
    return dict(task=task,fit_ids=dataset.ids[fit].tolist(),test_ids=dataset.ids[test].tolist(),
                targets=dataset.target_names,model_names=sorted(predictions),diagnostics=result['diagnostics'],predictions=rows)

def validate_payload(payload,dataset,cfg,split,task):
    if payload['task']!=task: raise ValueError('Task identity mismatch')
    fit,test=roles(dataset,split,task)
    if payload['fit_ids']!=dataset.ids[fit].tolist() or payload['test_ids']!=dataset.ids[test].tolist(): raise ValueError('Cached bottle-role mismatch')
    if set(payload['fit_ids']) & set(payload['test_ids']): raise ValueError('Test bottles leaked into fitting')
    if payload['targets']!=dataset.target_names: raise ValueError('Target order mismatch')
    models=expected_models(task['condition'],cfg)
    if set(payload['model_names'])!=set(models): raise ValueError('Cache model set mismatch')
    frame=pd.DataFrame(payload['predictions'])
    if len(frame)!=len(test)*11*len(models) or frame.duplicated(['model','sample_id','target']).any(): raise ValueError('Prediction keys are incomplete or duplicated')
    if set(frame.model)!=set(models) or set(frame.target)!=set(dataset.target_names) or set(frame.sample_id)!=set(dataset.ids[test]): raise ValueError('Prediction identities disagree')
    if not np.isfinite(frame[['y_true','y_pred','train_sd','outer_train_sd','effective_nir_weight','n_acquisition_modalities']].to_numpy(float)).all(): raise ValueError('Nonfinite cached prediction')
    Y=dataset.Y
    expected_truth=np.array([Y.loc[str(identity),target] for identity,target in zip(frame.sample_id,frame.target)],float)
    if not np.array_equal(frame.y_true.to_numpy(float),expected_truth): raise ValueError('Cached reference concentrations changed')
    train_sd=Y.iloc[fit].std(axis=0,ddof=1)
    outer_sd=Y.iloc[split['tr']].std(axis=0,ddof=1)
    if not np.allclose(frame.train_sd,[train_sd[t] for t in frame.target],rtol=0,atol=1e-12): raise ValueError('Cached fitting scales changed')
    if not np.allclose(frame.outer_train_sd,[outer_sd[t] for t in frame.target],rtol=0,atol=1e-12): raise ValueError('Cached outer scales changed')
    if not frame.n_train.eq(len(fit)).all() or not frame.n_unique_train.eq(len(set(dataset.ids[fit]))).all(): raise ValueError('Training counts changed')
    if not frame.n_cal.eq(0).all(): raise ValueError('Point tasks cannot contain calibration rows')
    for key,value in task.items():
        if key in frame and not frame[key].eq(value).all(): raise ValueError('Row task metadata mismatch: '+key)
    if ((frame.effective_nir_weight==0)&(frame.n_acquisition_modalities!=2)).any(): raise ValueError('Zero-weight acquisition count mismatch')
    if not frame.groupby('model').effective_nir_weight.nunique().eq(1).all(): raise ValueError('A shared gate must use one weight for the full target panel')
    if not frame.groupby('model').n_acquisition_modalities.nunique().eq(1).all(): raise ValueError('Acquisition requirements must describe the complete panel')
    selections=payload['diagnostics'].get('selections',{})
    acquisition=payload['diagnostics'].get('acquisition_by_model',{})
    if set(selections)!=set(models) or set(acquisition)!=set(models): raise ValueError('Model selection or acquisition records are incomplete')
    for model in models:
        records=selections[model]
        if len(records)!=11 or sorted(record['target_index'] for record in records)!=list(range(11)):
            raise ValueError('Every model requires eleven distinct target selection records')
        group=frame[frame.model.eq(model)]
        item=acquisition[model]
        modalities=item['modality_union']
        n_inputs=int(item['n_modalities'])
        if len(set(modalities))!=n_inputs or not group.n_acquisition_modalities.eq(n_inputs).all():
            raise ValueError('Prediction and acquisition diagnostics disagree')
        if model.endswith('__uv+raman') or model.endswith('__weight_zero') or model.endswith('__zero'):
            if set(modalities)!={'uv','raman'} or not group.effective_nir_weight.eq(0).all():
                raise ValueError('The zero-NIR baseline must use exactly U+R')
        expected_weights=[]
        for record in records:
            spec=record['spec']
            expected_weights.append(float(spec.get('nir_weight',spec.get('multiplier',spec.get('correction_multiplier',0.0 if 'nir' not in modalities else 1.0)))))
        if len(set(expected_weights))!=1 or not group.effective_nir_weight.eq(expected_weights[0]).all():
            raise ValueError('Selected shared weights and prediction metadata disagree')
        if set(modalities)!=( {'uv','raman'} if expected_weights[0]==0 else {'uv','raman','nir'} ):
            raise ValueError('NIR weight and required prediction modalities disagree')
    inner=payload['diagnostics'].get('inner_splits',[])
    if len(inner)!=int(cfg['inner_folds']): raise ValueError('Inner role records are incomplete')
    validation_counts=Counter()
    for item in inner:
        train=set(map(str,item['train_ids']))
        validation=set(map(str,item['validation_ids']))
        if train & validation or train | validation != set(payload['fit_ids']): raise ValueError('Inner bottle roles are invalid')
        if (train | validation) & set(payload['test_ids']): raise ValueError('Outer evaluation identities appear in inner roles')
        validation_counts.update(validation)
    if set(validation_counts)!=set(payload['fit_ids']) or any(value!=1 for value in validation_counts.values()):
        raise ValueError('Each training bottle must appear in exactly one inner validation fold')
    return True

def checked_cache(path,signature,dataset,cfg,split,task):
    record=read_json(path)
    if record.get('signature')!=signature: raise ValueError('Cache signature mismatch: '+str(path))
    if record.get('payload_sha256')!=digest_object(record['payload']): raise ValueError('Cache payload hash mismatch: '+str(path))
    validate_payload(record['payload'],dataset,cfg,split,task)
    return record['payload']

def consolidate(out,plan,signature,dataset,cfg,lookup):
    fields=['stage','condition','repeat','fold','training_size','budget_refit','bootstrap_refit','seed','model','sample_id','target','y_true','y_pred','train_sd','outer_train_sd','n_train','n_unique_train','n_cal','effective_nir_weight','n_acquisition_modalities']
    temporary=out/'predictions.csv.gz.tmp'
    finished=[]
    count=0
    with gzip.open(str(temporary),'wt',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for task in plan:
            path=out/'tasks'/task_name(task)
            if not path.exists(): continue
            payload=checked_cache(path,signature,dataset,cfg,lookup[(task['repeat'],task['fold'])],task)
            writer.writerows(payload['predictions'])
            count+=len(payload['predictions'])
            finished.append(task)
    os.replace(str(temporary),str(out/'predictions.csv.gz'))
    planned=Counter(t['condition'] for t in plan)
    completed=Counter(t['condition'] for t in finished)
    report=dict(status='all_requested_stages_complete' if len(finished)==len(plan) else 'incomplete',
                signature=signature,expected_tasks=len(plan),completed_tasks=len(finished),prediction_rows=count,
                stages={c:dict(expected_tasks=planned[c],completed_tasks=completed[c],complete=planned[c]==completed[c]) for c in planned},
                all_eleven_targets_retained=True,independent_bottles=len(dataset.ids),external_validation=False,
                profile=cfg['profile'],quick_results_are_formal_evidence=False)
    atomic_json(out/'COMPLETENESS.json',report)
    return report

@contextlib.contextmanager
def output_lock(out):
    handle=(out/'RUN_LOCK').open('a+b')
    try:
        handle.seek(0)
        if not handle.read(1):
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except (OSError,IOError):
        handle.close()
        raise RuntimeError('Another process is using this output folder; wait for it to finish')
    try:
        yield
    finally:
        handle.close()

def main(argv=None):
    parser=argparse.ArgumentParser(description='Run the V8 retrospective NIR information-value extension')
    parser.add_argument('--profile',choices=('full','quick'),default='full')
    parser.add_argument('--stage',choices=('all','prepare','summarize','archived')+CONDITIONS,default='all')
    parser.add_argument('--out')
    parser.add_argument('--max-tasks',type=int)
    args=parser.parse_args(argv)
    cfg=read_json(ROOT/'configs'/('v8_'+args.profile+'.json'))
    out=Path(args.out).resolve() if args.out else ROOT/'runs'/('v8_'+args.profile)
    protected=[ROOT.resolve()]+[(ROOT/name).resolve() for name in ('results','executed','data','figures','provenance','v8','configs','tools','docs')]
    if out==ROOT.resolve() or any(out==p or p in out.parents for p in protected[1:]):
        raise ValueError('Release sources and recorded results are immutable; use runs or work')
    out.mkdir(parents=True,exist_ok=True)
    with output_lock(out):
        return run_study(args,cfg,out)

def run_study(args,cfg,out):
    dataset=load_dataset(ROOT/'executed/data')
    if len(dataset.ids)!=139 or len(dataset.target_names)!=11: raise ValueError('The release requires all 139 bottles and eleven targets')
    splits=historical_splits(dataset.ids,cfg)
    plan=plan_tasks(cfg,splits)
    signature,context,environment=execution_context(cfg,dataset)
    previous=out/'protocol.json'
    if previous.exists() and read_json(previous).get('signature')!=signature:
        raise ValueError('Source/configuration/runtime changed. Choose a new output folder; existing runs are preserved.')
    atomic_json(previous,dict(signature=signature,context=context))
    atomic_json(out/'environment.json',environment)
    if environment.get('pip_freeze') is not None: (out/'environment.freeze.txt').write_text(environment['pip_freeze'],encoding='utf-8')
    atomic_json(out/'task_plan.json',plan)
    atomic_json(out/'data_audit.json',dataset.audit)
    atomic_json(out/'splits.json',[{k:v for k,v in s.items() if k not in ('tr','te')} for s in splits])
    (out/'tasks').mkdir(exist_ok=True)
    lookup={(s['repeat'],s['fold']):s for s in splits}
    from v8.analysis import analyze_archive,analyze_run
    if args.stage=='archived':
        print(json.dumps(analyze_archive(ROOT,out,cfg),indent=2))
        return
    selected=[t for t in plan if args.stage=='all' or t['condition']==args.stage]
    if args.stage in ('prepare','summarize'): selected=[]
    executed=0
    for index,task in enumerate(selected,1):
        path=out/'tasks'/task_name(task)
        if path.exists():
            checked_cache(path,signature,dataset,cfg,lookup[(task['repeat'],task['fold'])],task)
            print('[%d/%d] verified cached %s'%(index,len(selected),path.name),flush=True)
            continue
        if args.max_tasks is not None and executed>=args.max_tasks: break
        print('[%d/%d] running %s'%(index,len(selected),path.name),flush=True)
        started=time.time()
        payload=evaluate(dataset,cfg,lookup[(task['repeat'],task['fold'])],task)
        validate_payload(payload,dataset,cfg,lookup[(task['repeat'],task['fold'])],task)
        atomic_json(path,dict(signature=signature,payload_sha256=digest_object(payload),payload=payload))
        executed+=1
        print('completed %.1f seconds; %d prediction rows'%(time.time()-started,len(payload['predictions'])),flush=True)
    report=consolidate(out,plan,signature,dataset,cfg,lookup)
    if args.stage in ('all','summarize') and report['completed_tasks']:
        analyze_run(ROOT,out,cfg)
    if args.stage=='all' and report['status']=='all_requested_stages_complete':
        analyze_archive(ROOT,out/'archived_comparison',cfg)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    main()
