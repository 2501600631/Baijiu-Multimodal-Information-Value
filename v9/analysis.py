from pathlib import Path
import gzip
import hashlib
import json
import numpy as np
import pandas as pd
from v8.analysis import TARGETS, PAIR_KEY, bootstrap_draws, conditional_pair, corrected_cv, holm_adjust, normalize_predictions, panel_metrics, read_predictions, target_metrics, validate_complete_pair

MASKS=('uv+raman','uv+raman+nir')
METHODS=('pls','ridge','krr','sopls','sopls_perblock')
METHOD_LABELS={'pls':'pls','ridge':'ridge','krr':'krr','sopls':'sopls_uniform','sopls_perblock':'sopls_perblock'}
PRIMARY='primary_five_method_nir_addition'
SECONDARY='secondary_uniform_vs_perblock_fixed_mask'
CI_SCOPE='Conditional paired bottle bootstrap of frozen OOF losses; no refitting or simultaneous population-coverage claim'

def _json(path):
    path=Path(path)
    with (gzip.open(str(path),'rt',encoding='utf-8') if path.suffix=='.gz' else path.open(encoding='utf-8')) as stream:
        return json.load(stream)

def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):
            digest.update(block)
    return digest.hexdigest()

def _safe(value):
    if isinstance(value,dict):
        return {str(key):_safe(item) for key,item in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):
        return [_safe(item) for item in value]
    if isinstance(value,(np.integer,np.bool_)):
        return value.item()
    if isinstance(value,(float,np.floating)):
        return float(value) if np.isfinite(value) else None
    return value

def _dump(path,value):
    Path(path).write_text(json.dumps(_safe(value),indent=2,ensure_ascii=True,allow_nan=False)+'\n',encoding='utf-8')

def _comparison_specs():
    result=[]
    for method in METHODS:
        result.append(dict(family=PRIMARY,method=METHOD_LABELS[method],comparison='uv+raman__to__uv+raman+nir',
                           before_model=method+'__'+MASKS[0],after_model=method+'__'+MASKS[1]))
    for mask in MASKS:
        result.append(dict(family=SECONDARY,method='sopls_perblock_vs_uniform',comparison=mask,
                           before_model='sopls__'+mask,after_model='sopls_perblock__'+mask))
    return result

def _paired_losses(before,after,expected_ids,require_full,split_records):
    before,after=validate_complete_pair(before,after,expected_ids,split_records=split_records,require_full=require_full)
    ids=sorted(before.sample_id.unique())
    arrays=[]
    for frame in (before,after):
        grouped=frame.groupby(['sample_id','target'],sort=False)[['squared_error','standardized_squared_error']].mean()
        raw=grouped.squared_error.unstack('target').reindex(index=ids,columns=TARGETS).to_numpy(float)
        standardized=grouped.standardized_squared_error.unstack('target').reindex(index=ids,columns=TARGETS).to_numpy(float)
        arrays.append((raw,standardized))
    return before,after,ids,arrays

def _task_effects(before,after,meta):
    keys=['repeat','fold','target']
    a=before.groupby(keys,sort=False).agg(raw_before=('squared_error','mean'),mse_before=('standardized_squared_error','mean'),
                                       n_test=('sample_id','nunique'),n_train=('n_train','first')).reset_index()
    b=after.groupby(keys,sort=False).agg(raw_after=('squared_error','mean'),mse_after=('standardized_squared_error','mean')).reset_index()
    joined=a.merge(b,on=keys,validate='one_to_one')
    joined['raw_difference']=joined.raw_after-joined.raw_before
    joined['standardized_difference']=joined.mse_after-joined.mse_before
    panels=[]
    for (repeat,fold),group in joined.groupby(['repeat','fold'],sort=False):
        if set(group.target)!=set(TARGETS) or len(group)!=11:
            raise ValueError('Each paired task must retain eleven targets')
        mb,ma=float(group.mse_before.mean()),float(group.mse_after.mean())
        panels.append(dict(meta,repeat=int(repeat),fold=int(fold),n_targets=11,n_test=int(group.n_test.iloc[0]),
                           n_train=int(group.n_train.iloc[0]),macro_mse_before=mb,macro_mse_after=ma,
                           delta_mse_after_minus_before=ma-mb,gain_pct=100*(1-ma/mb) if mb>0 else np.nan))
    return pd.DataFrame(panels),joined

def _adjust_family(table,panel):
    if table.empty:
        return table
    table=table.copy()
    table['p_holm']=np.nan
    table['planned_family_size']=0
    table['multiplicity_scope']=''
    for family,size in ((PRIMARY,5 if panel else 55),(SECONDARY,2 if panel else 22)):
        positions=table.index[table.family.eq(family)]
        if len(positions)!=size:
            raise ValueError('The complete declared Holm family is required: '+family)
        table.loc[positions,'p_holm']=holm_adjust(table.loc[positions,'p_uncorrected'].to_numpy(float))
        table.loc[positions,'planned_family_size']=size
        table.loc[positions,'multiplicity_scope']=family
    return table

def compute_comparisons(frame,expected_ids,cfg,complete,split_records=None):
    frame=normalize_predictions(frame)
    available=set(frame.model)
    target_effects,panel_effects,task_effects,target_ci,panel_ci,target_cv,panel_cv,statuses=[],[],[],[],[],[],[],[]
    draws=int(cfg.get('statistical_bootstrap_draws',2000))
    seed=int(cfg.get('statistics_seed',20261104))
    indices=bootstrap_draws(len(expected_ids),draws,seed) if complete else None
    new=frame[frame.model.str.startswith('sopls_perblock__')]
    if new.empty:
        scope_keys=pd.DataFrame(columns=PAIR_KEY)
    else:
        scope_keys=new[PAIR_KEY].drop_duplicates()
    for meta in _comparison_specs():
        if meta['before_model'] not in available or meta['after_model'] not in available:
            statuses.append(dict(meta,status='missing_paired_model',conditional_intervals=False))
            continue
        before=frame[frame.model.eq(meta['before_model'])]
        after=frame[frame.model.eq(meta['after_model'])]
        if not complete:
            before=before.merge(scope_keys,on=PAIR_KEY,validate='one_to_one')
            after=after.merge(scope_keys,on=PAIR_KEY,validate='one_to_one')
        if before.empty or after.empty:
            statuses.append(dict(meta,status='no_new_completed_tasks',conditional_intervals=False))
            continue
        before,after,ids,arrays=_paired_losses(before,after,expected_ids,complete,split_records)
        raw_b,std_b=arrays[0]
        raw_a,std_a=arrays[1]
        rb,ra=np.sqrt(raw_b.mean(axis=0)),np.sqrt(raw_a.mean(axis=0))
        mb,ma=float(std_b.mean()),float(std_a.mean())
        for j,target in enumerate(TARGETS):
            target_effects.append(dict(meta,target=target,n_unique_bottles=len(ids),n_prediction_visits=len(before)//11,
                                       rmse_before=float(rb[j]),rmse_after=float(ra[j]),rmse_difference_after_minus_before=float(ra[j]-rb[j]),
                                       gain_pct=100*(1-ra[j]/rb[j]) if rb[j]>0 else np.nan,
                                       analysis_status='complete' if complete else 'partial_descriptive_only'))
        panel_effects.append(dict(meta,n_unique_bottles=len(ids),n_prediction_visits=len(before)//11,n_targets=11,
                                  macro_mse_before=mb,macro_mse_after=ma,delta_mse_after_minus_before=ma-mb,
                                  gain_pct=100*(1-ma/mb) if mb>0 else np.nan,
                                  analysis_status='complete' if complete else 'partial_descriptive_only'))
        tasks,joined=_task_effects(before,after,meta)
        task_effects.extend(tasks.to_dict('records'))
        if not complete:
            statuses.append(dict(meta,status='partial_descriptive_only',conditional_intervals=False))
            continue
        result=conditional_pair(raw_b,raw_a,std_b,std_a,indices)
        for j,target in enumerate(TARGETS):
            target_ci.append(dict(meta,target=target,n_unique_bottles=len(ids),n_prediction_visits=len(before)//11,
                                  rmse_before=result['rmse_before'][j],rmse_after=result['rmse_after'][j],
                                  gain_pct=result['target_gain_pct'][j],conditional_ci_low_pct=result['target_gain_bounds'][0,j],
                                  conditional_ci_high_pct=result['target_gain_bounds'][1,j],
                                  rmse_difference_after_minus_before=result['target_rmse_difference_after_minus_before'][j],
                                  rmse_difference_ci_low=result['target_rmse_difference_bounds'][0,j],
                                  rmse_difference_ci_high=result['target_rmse_difference_bounds'][1,j],
                                  bootstrap_draws=draws,bootstrap_seed=seed,interval_scope=CI_SCOPE))
            group=joined[joined.target.eq(target)]
            target_cv.append(dict(meta,target=target,loss='raw_unit_mean_squared_error',
                                  **corrected_cv(group.raw_difference.to_numpy(float),(group.n_test/group.n_train).to_numpy(float))))
        panel_ci.append(dict(meta,n_unique_bottles=len(ids),n_targets=11,macro_mse_before=result['macro_mse_before'],
                             macro_mse_after=result['macro_mse_after'],delta_mse_after_minus_before=result['macro_mse_difference_after_minus_before'],
                             conditional_delta_ci_low=result['macro_mse_difference_bounds'][0],conditional_delta_ci_high=result['macro_mse_difference_bounds'][1],
                             gain_pct=result['macro_gain_pct'],conditional_gain_ci_low_pct=result['macro_gain_bounds'][0],
                             conditional_gain_ci_high_pct=result['macro_gain_bounds'][1],bootstrap_draws=draws,bootstrap_seed=seed,interval_scope=CI_SCOPE))
        by_task=joined.groupby(['repeat','fold'],sort=False).agg(difference=('standardized_difference','mean'),n_test=('n_test','first'),n_train=('n_train','first'))
        panel_cv.append(dict(meta,loss='equal_target_standardized_mean_squared_error',
                             **corrected_cv(by_task.difference.to_numpy(float),(by_task.n_test/by_task.n_train).to_numpy(float))))
        statuses.append(dict(meta,status='complete_paired_conditional_analysis',conditional_intervals=True))
    target_cv=pd.DataFrame(target_cv)
    panel_cv=pd.DataFrame(panel_cv)
    if complete:
        target_cv=_adjust_family(target_cv,False)
        panel_cv=_adjust_family(panel_cv,True)
    columns=['family','method','comparison','before_model','after_model']
    return dict(paired_target_effects=pd.DataFrame(target_effects),paired_panel_effects=pd.DataFrame(panel_effects),
                paired_task_comparison=pd.DataFrame(task_effects),target_conditional_intervals=pd.DataFrame(target_ci,columns=None if target_ci else columns+['target','gain_pct','conditional_ci_low_pct','conditional_ci_high_pct']),
                panel_conditional_intervals=pd.DataFrame(panel_ci,columns=None if panel_ci else columns+['gain_pct','conditional_gain_ci_low_pct','conditional_gain_ci_high_pct']),
                target_corrected_cv_joined_family=target_cv if not target_cv.empty else pd.DataFrame(columns=columns+['target','p_uncorrected','p_holm','planned_family_size']),
                panel_corrected_cv_joined_family=panel_cv if not panel_cv.empty else pd.DataFrame(columns=columns+['p_uncorrected','p_holm','planned_family_size']),
                comparison_status=pd.DataFrame(statuses),draw_metadata=dict(draws=draws,seed=seed,
                    draw_indices_sha256=hashlib.sha256(np.ascontiguousarray(indices).tobytes()).hexdigest() if complete else None,
                    scope=CI_SCOPE,all_targets_and_models_share_indices=True))

def _details_for_selection(diagnostics,model,record,index):
    for name in ('final_fit_details','final_details','fit_details'):
        if isinstance(record.get(name),dict):
            return record[name]
    fits=diagnostics.get('final_fits',{})
    for key in (str(record.get('candidate_id')),str(record.get('path_id')),model):
        item=fits.get(key,{}) if isinstance(fits,dict) else {}
        if isinstance(item,list):
            matching=[entry for entry in item if int(entry.get('target_index',-1))==index]
            if len(matching)==1 and isinstance(matching[0].get('details'),dict):
                return matching[0]['details']
        if isinstance(item,dict):
            effective=item.get('effective_by_target',{})
            if str(index) in effective:
                return effective[str(index)]
            if index in effective:
                return effective[index]
    return {}

def selection_table(payloads,source):
    rows=[]
    for payload in payloads:
        task=payload.get('task',payload)
        repeat,fold=int(task['repeat']),int(task['fold'])
        diagnostics=payload['diagnostics']
        for model,records in diagnostics['selections'].items():
            if model not in {method+'__'+mask for method in ('sopls','sopls_perblock') for mask in MASKS}:
                continue
            if len(records)!=11 or {int(item['target_index']) for item in records}!=set(range(11)):
                raise ValueError('SO-PLS selection records must retain eleven distinct targets')
            for item in records:
                j=int(item['target_index'])
                spec=item['spec']
                order=list(spec['order'])
                components=spec.get('components_by_block',spec.get('components'))
                if np.isscalar(components):
                    components=[int(components)]*len(order)
                else:
                    components=[int(value) for value in components]
                if len(components)!=len(order):
                    raise ValueError('Ordered component vector and block order disagree')
                details=_details_for_selection(diagnostics,model,item,j)
                active={block['modality']:int(block['components_active']) for block in details.get('blocks',[])}
                by_modality=dict(zip(order,components))
                record=dict(source=source,repeat=repeat,fold=fold,model=model,target=TARGETS[j],target_index=j,
                            block_order='+'.join(order),k1=components[0],k2=components[1],k3=components[2] if len(components)>2 else np.nan,
                            k_uv=by_modality.get('uv',np.nan),k_raman=by_modality.get('raman',np.nan),k_nir=by_modality.get('nir',np.nan),
                            requested_total_components=sum(components),active_total_components=details.get('total_components_active',np.nan),
                            active_k1=active.get(order[0],np.nan),active_k2=active.get(order[1],np.nan),active_k3=active.get(order[2],np.nan) if len(order)>2 else np.nan,
                            inner_mse=item.get('inner_mse',np.nan),inner_standardized_mse=item.get('inner_standardized_mse',np.nan),
                            preprocessing=json.dumps(spec['preprocessing'],sort_keys=True,separators=(',',':')),
                            components_scope='Ordered vector; k1/k2/k3 follow block_order and are target-specific')
                stages={int(stage['stage']):stage for stage in item.get('stage_inner_scores',[])}
                for position in range(1,4):
                    stage=stages.get(position,stages.get(position-1,{}) if 0 in stages else {})
                    record['stage%d_inner_mse'%position]=stage.get('inner_mse',np.nan)
                rows.append(record)
    return pd.DataFrame(rows)

def selected_stage_scores(payloads):
    rows=[]
    for payload in payloads:
        task=payload.get('task',payload)
        for model,records in payload['diagnostics']['selections'].items():
            for record in records:
                for stage in record.get('stage_inner_scores',[]):
                    rows.append(dict(repeat=int(task['repeat']),fold=int(task['fold']),model=model,
                                     target=TARGETS[int(record['target_index'])],target_index=int(record['target_index']),
                                     block_order='+'.join(record['spec']['order']),**stage))
    return pd.DataFrame(rows,columns=None if rows else ['repeat','fold','model','target','target_index','block_order','stage','modality','selected_k','inner_mse'])

def paired_task_parameters(selections,task_effects):
    parameters=selections.copy()
    parameters['method']=parameters.model.str.split('__',n=1).str[0].map({'sopls':'sopls_uniform','sopls_perblock':'sopls_perblock'})
    parameters['mask']=parameters.model.str.split('__',n=1).str[1]
    keys=['repeat','fold','method','target','target_index']
    values=['block_order','k1','k2','k3','k_uv','k_raman','k_nir','requested_total_components','active_total_components',
            'active_k1','active_k2','active_k3','inner_mse','inner_standardized_mse','preprocessing','stage1_inner_mse','stage2_inner_mse','stage3_inner_mse']
    tables=[]
    for mask,prefix in ((MASKS[0],'ur_'),(MASKS[1],'urn_')):
        part=parameters[parameters['mask'].eq(mask)][keys+values].copy()
        part=part.rename(columns={name:prefix+('order' if name=='block_order' else name) for name in values})
        tables.append(part)
    joined=tables[0].merge(tables[1],on=keys,validate='one_to_one')
    panel=task_effects[task_effects.family.eq(PRIMARY)&task_effects.method.isin(['sopls_uniform','sopls_perblock'])].copy()
    joined=joined.merge(panel,on=['repeat','fold','method'],validate='many_to_one')
    joined['panel_metric_scope']='Repeated for each of eleven target parameter records; count independent outer tasks using repeat/fold/method, not rows'
    return joined

def _payloads(folder,pattern):
    for path in sorted(Path(folder).glob(pattern)):
        wrapper=_json(path)
        yield wrapper.get('payload',wrapper)

def analyze(root,run_dir,cfg):
    root,run_dir=Path(root),Path(run_dir)
    derived=run_dir/'derived'
    derived.mkdir(parents=True,exist_ok=True)
    ids=pd.read_csv(root/'executed/data/chemistry.csv',dtype={'sample_id':str},usecols=['sample_id']).sample_id.tolist()
    source=run_dir/'predictions.csv.gz'
    new=read_predictions(source)
    expected_new={'sopls_perblock__'+mask for mask in MASKS}
    if set(new.model)!=expected_new or set(new.condition)!={'sopls_perblock'}:
        raise ValueError('The extension requires precisely the two declared per-block model masks')
    old_source=root/'results/baseline/benchmark/predictions.csv.gz'
    old=read_predictions(old_source)
    wanted={method+'__'+mask for method in METHODS[:-1] for mask in MASKS}
    old=old[old.model.isin(wanted)].copy()
    if set(old.model)!=wanted:
        raise ValueError('All four historical paired algorithm families must remain available')
    combined=pd.concat([old,new],ignore_index=True,sort=False)
    completeness=_json(run_dir/'COMPLETENESS.json')
    new_tasks=new[['repeat','fold']].drop_duplicates()
    complete=len(new_tasks)==15 and int(completeness.get('extension_completed_tasks',0))==15
    splits=_json(run_dir/'splits.json') if (run_dir/'splits.json').exists() else None
    if isinstance(splits,dict):
        splits=splits.get('splits')
    if complete:
        for model in expected_new:
            context=new[new.model.eq(model)]
            validate_complete_pair(context,context,ids,split_records=splits,require_full=True)
    comparisons=compute_comparisons(combined,ids,cfg,complete,splits)
    fold_target=target_metrics(combined,['condition','model','repeat','fold','target'])
    fold_panel=panel_metrics(fold_target,['condition','model','repeat','fold'])
    pooled=target_metrics(combined,['condition','model','target'])
    new_selections=[]
    new_stages=[]
    for payload in _payloads(run_dir/'tasks','*.json.gz'):
        new_selections.append(selection_table([payload],'new_perblock'))
        new_stages.append(selected_stage_scores([payload]))
    old_selections=selection_table(_payloads(root/'results/baseline/tasks','benchmark_*.json.gz'),'historical_uniform')
    selections=pd.concat([old_selections]+new_selections,ignore_index=True,sort=False)
    stages=pd.concat(new_stages,ignore_index=True,sort=False) if new_stages else selected_stage_scores([])
    table_tasks=comparisons['paired_task_comparison']
    tables=dict(fold_target_metrics=fold_target,fold_panel_metrics=fold_panel,pooled_target_metrics=pooled,
                parameter_selection_by_task_target=selections,
                selected_stage_scores=stages,
                uniform_perblock_paired_task_parameters=paired_task_parameters(selections,table_tasks),
                uniform_perblock_task_comparison=table_tasks[table_tasks.family.eq(PRIMARY)&table_tasks.method.isin(['sopls_uniform','sopls_perblock'])].copy(),
                secondary_fixed_mask_task_comparison=table_tasks[table_tasks.family.eq(SECONDARY)].copy())
    tables.update({name:value for name,value in comparisons.items() if isinstance(value,pd.DataFrame)})
    written={}
    for name,frame in tables.items():
        destination=derived/(name+'.csv')
        frame.to_csv(destination,index=False,float_format='%.17g')
        written[name]=dict(path='derived/'+destination.name,rows=len(frame),sha256=_sha(destination))
    registry=dict(status='complete_extension_postprocessed' if complete else 'partial_descriptive_only',
                  completed_extension_tasks=len(new_tasks),expected_extension_tasks=15,n_independent_bottles=len(ids),n_targets=11,
                  source=source.name,source_sha256=_sha(source),historical_source=old_source.relative_to(root).as_posix(),historical_source_sha256=_sha(old_source),
                  analysis_source_sha256=_sha(Path(__file__)),model_fits_performed=0,tables=written,
                  primary_holm_family=dict(panel=5,target=55,methods=list(METHOD_LABELS.values()),contrast='U+R to U+R+N'),
                  secondary_holm_family=dict(panel=2,target=22,contrast='Uniform to per-block within U+R and U+R+N'),
                  conditional_bootstrap=comparisons['draw_metadata'],historical_tables_overwritten=False,
                  structural_scope='Greedy per-block allocation with component grid 1,2,3,5 versus historical uniform grid 1,2,3,5,10; not an isolated upper-bound expansion',
                  inference_scope='Retrospective same-cohort structural extension; conditional intervals and approximate corrected-CV sensitivity; all directions retained')
    _dump(run_dir/'analysis_registry.json',registry)
    return registry
