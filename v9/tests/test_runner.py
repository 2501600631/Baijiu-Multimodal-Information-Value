from pathlib import Path
from collections import Counter
import copy
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from v9 import runner

TARGETS=['acetaldehyde','methanol','n_propanol','ethyl_acetate','isobutanol','acetal','n_butanol','isoamyl_alcohol','ethyl_butyrate','ethyl_lactate','ethyl_hexanoate']

def synthetic_dataset(n=36):
    rng=np.random.RandomState(491)
    ids=np.array(['V9B%03d'%i for i in range(n)])
    blocks={m:rng.normal(size=(n,8)) for m in ('uv','raman','nir')}
    latent=np.column_stack([blocks[m][:,0] for m in ('uv','raman','nir')])
    values=latent.dot(rng.normal(size=(3,11)))+rng.normal(scale=.4,size=(n,11))+np.arange(11)
    response=pd.DataFrame(values,index=ids,columns=TARGETS)
    axes={m:np.linspace(1,10,8) for m in blocks}
    return SimpleNamespace(ids=ids,X=blocks,Y=response,axes=axes,target_names=TARGETS,audit={})

def config():
    return dict(seed=20261029,inner_folds=3,repeats=3,outer_folds=5,
                perblock=dict(components=[1,2,3,5]),sopls_preprocessing_options={m:['raw'] for m in ('uv','raman','nir')},
                preprocessing_settings={})

def splits(n=139):
    output=[]
    partitions=np.array_split(np.arange(n),5)
    for repeat in range(3):
        for fold,test in enumerate(partitions):
            output.append(dict(repeat=repeat,fold=fold,tr=np.setdiff1d(np.arange(n),test),te=test))
    return output

def parent_status():
    return dict(status='all_requested_stages_complete',completed_tasks=255,expected_tasks=255,prediction_rows=376134,
                stages={name:dict(expected_tasks=count,completed_tasks=count,complete=True) for name,count in
                        [('unscaled',15),('nir_extended',15),('nir_intervals',15),('balanced_budget',75),('nir_weight',15),('residual',15),('nir_weight_learning',30),('nir_weight_stability',75)]})

class RunnerChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=synthetic_dataset()
        cls.cfg=config()
        cls.split=dict(repeat=0,fold=0,tr=np.arange(27),te=np.arange(27,36))
        cls.task=runner.task_plan(cls.cfg,[cls.split])[0]
        cls.payload=runner.evaluate(cls.data,cls.cfg,cls.split,cls.task)

    def validate(self,payload):
        return runner.validate_payload(payload,self.data,self.cfg,self.split,self.task)

    def test_complete_synthetic_two_mask_eleven_target_roundtrip(self):
        self.assertTrue(self.validate(self.payload))
        frame=pd.DataFrame(self.payload['predictions'])
        self.assertEqual(len(frame),9*11*2)
        self.assertEqual(set(frame.target),set(TARGETS))
        self.assertEqual(set(frame.model),set(runner.MODELS))
        self.assertFalse(set(self.payload['fit_ids'])&set(self.payload['test_ids']))
        for model in runner.MODELS:
            group=frame[frame.model.eq(model)]
            self.assertTrue(group.n_acquisition_modalities.eq(3 if model.endswith('+nir') else 2).all())
            self.assertTrue(group.effective_nir_weight.eq(float(model.endswith('+nir'))).all())

    def test_plan_has_exactly_fifteen_unique_prespecified_tasks(self):
        plan=runner.task_plan(config(),splits())
        self.assertEqual(len(plan),15)
        self.assertEqual(len({runner.task_name(task) for task in plan}),15)
        self.assertEqual(Counter(task['repeat'] for task in plan),Counter({0:5,1:5,2:5}))
        for task in plan:
            self.assertEqual(task['condition'],'sopls_perblock')
            self.assertEqual(task['seed'],20261029+10000*task['repeat']+100*task['fold'])
            self.assertEqual(task['training_size'],'full')
            self.assertEqual(task['bootstrap_refit'],-1)

    def test_outer_evaluation_labels_do_not_affect_fitting_or_selection(self):
        data=copy.deepcopy(self.data)
        data.Y.iloc[self.split['te']]=data.Y.iloc[self.split['te']]*100+999
        second=runner.evaluate(data,self.cfg,self.split,self.task)
        first=pd.DataFrame(self.payload['predictions'])
        changed=pd.DataFrame(second['predictions'])
        np.testing.assert_array_equal(first.y_pred.to_numpy(),changed.y_pred.to_numpy())
        self.assertEqual(self.payload['diagnostics']['selections'],second['diagnostics']['selections'])
        self.assertFalse(np.array_equal(first.y_true.to_numpy(),changed.y_true.to_numpy()))

    def test_truth_scales_and_task_metadata_corruption_are_rejected(self):
        for field in ('y_true','train_sd','outer_train_sd','n_cal','n_train','n_unique_train','seed'):
            payload=copy.deepcopy(self.payload)
            payload['predictions'][0][field]+=1
            with self.subTest(field=field),self.assertRaises(ValueError):
                self.validate(payload)

    def test_bottle_role_target_order_and_prediction_key_corruption_are_rejected(self):
        payload=copy.deepcopy(self.payload)
        payload['fit_ids'][0]=payload['test_ids'][0]
        with self.assertRaises(ValueError): self.validate(payload)
        payload=copy.deepcopy(self.payload)
        payload['targets']=payload['targets'][::-1]
        with self.assertRaises(ValueError): self.validate(payload)
        payload=copy.deepcopy(self.payload)
        payload['predictions'][-1]=copy.deepcopy(payload['predictions'][0])
        with self.assertRaises(ValueError): self.validate(payload)

    def test_component_vector_alias_map_and_fitted_vector_corruption_are_rejected(self):
        for mutation in ('valid_vector','out_of_grid','alias','map','fitted_vector'):
            payload=copy.deepcopy(self.payload)
            model=runner.MODELS[0]
            spec=payload['diagnostics']['selections'][model][0]['spec']
            if mutation=='valid_vector': spec['components_by_block'][0]=2 if spec['components_by_block'][0]!=2 else 3
            if mutation=='out_of_grid': spec['components_by_block'][0]=10
            if mutation=='alias': spec['components']=[5,5]
            if mutation=='map': spec['component_map'][spec['order'][0]]=10
            if mutation=='fitted_vector': payload['diagnostics']['final_fits'][model][0]['details']['blocks'][0]['requested_components']=10
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                self.validate(payload)

    def test_order_and_preprocessing_cannot_be_detached_from_final_fit(self):
        for mutation in ('selected_order','fitted_order','selected_preprocessing'):
            payload=copy.deepcopy(self.payload)
            model=runner.MODELS[0]
            spec=payload['diagnostics']['selections'][model][0]['spec']
            if mutation=='selected_order': spec['order']=spec['order'][::-1]
            if mutation=='fitted_order': payload['diagnostics']['final_fits'][model][0]['details']['block_order'].reverse()
            if mutation=='selected_preprocessing': spec['preprocessing']['raman']='snv'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                self.validate(payload)

    def test_stage_score_component_or_stage_order_corruption_is_rejected(self):
        for mutation in ('k','order','count','final_score'):
            payload=copy.deepcopy(self.payload)
            selected=payload['diagnostics']['selections'][runner.MODELS[0]][0]
            stages=selected['stage_inner_scores']
            if mutation=='k': stages[0]['selected_k']=10
            if mutation=='order': stages[0]['modality']='nir'
            if mutation=='count': stages.pop()
            if mutation=='final_score': stages[-1]['inner_mse']+=1
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                self.validate(payload)

    def test_inner_fold_roles_row_indices_and_response_scales_are_bound(self):
        for mutation in ('outer_identity','repeat_fold','row_index','response_scale'):
            payload=copy.deepcopy(self.payload)
            inner=payload['diagnostics']['inner_splits']
            if mutation=='outer_identity': inner[0]['validation_ids'].append(payload['test_ids'][0])
            if mutation=='repeat_fold': inner[1]=copy.deepcopy(inner[0])
            if mutation=='row_index': inner[0]['train_row_indices'][0]=999
            if mutation=='response_scale': inner[0]['training_response_sd'][0]+=1
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                self.validate(payload)

    def test_fixed_mask_acquisition_counts_cannot_be_relabelled(self):
        payload=copy.deepcopy(self.payload)
        for row in payload['predictions']:
            if row['model']==runner.MODELS[0]:
                row['n_acquisition_modalities']=3
                row['effective_nir_weight']=1
        with self.assertRaises(ValueError): self.validate(payload)

    def test_cache_hash_signature_and_semantic_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix='_synthetic_v9_cache_',dir=str(ROOT)) as folder:
            path=Path(folder)/'task.json.gz'
            wrapper=dict(signature='synthetic-v9',payload_sha256=runner.base.digest_object(self.payload),payload=self.payload)
            runner.base.atomic_json(path,wrapper)
            checked=runner.checked_cache(path,'synthetic-v9',self.data,self.cfg,self.split,self.task)
            self.assertEqual(checked['fit_ids'],self.payload['fit_ids'])
            with self.assertRaisesRegex(ValueError,'signature'):
                runner.checked_cache(path,'wrong',self.data,self.cfg,self.split,self.task)
            altered=copy.deepcopy(wrapper)
            altered['payload']['predictions'][0]['y_pred']+=1
            runner.base.atomic_json(path,altered)
            with self.assertRaisesRegex(ValueError,'hash'):
                runner.checked_cache(path,'synthetic-v9',self.data,self.cfg,self.split,self.task)
            altered['payload']['predictions'][0]['y_true']+=1
            altered['payload_sha256']=runner.base.digest_object(altered['payload'])
            runner.base.atomic_json(path,altered)
            with self.assertRaisesRegex(ValueError,'reference'):
                runner.checked_cache(path,'synthetic-v9',self.data,self.cfg,self.split,self.task)

    def test_partial_and_complete_status_counts_parent_plus_extension(self):
        plan=runner.task_plan(config(),splits())
        lookup={(item['repeat'],item['fold']):self.split for item in plan}
        with tempfile.TemporaryDirectory(prefix='_synthetic_v9_status_',dir=str(ROOT)) as folder:
            out=Path(folder)
            (out/'tasks').mkdir()
            report=runner.consolidate(out,plan,'synthetic-v9',self.data,self.cfg,lookup,parent_status())
            self.assertEqual(report['expected_tasks'],270)
            self.assertEqual(report['completed_tasks'],255)
            self.assertEqual(report['extension_completed_tasks'],0)
            self.assertEqual(report['status'],'incomplete')
            for number,task in enumerate(plan):
                (out/'tasks'/runner.task_name(task)).write_text('synthetic cache placeholder')
                if number==0:
                    with patch.object(runner,'checked_cache',return_value=self.payload):
                        partial=runner.consolidate(out,plan,'synthetic-v9',self.data,self.cfg,lookup,parent_status())
                    self.assertEqual(partial['completed_tasks'],256)
                    self.assertEqual(partial['extension_completed_tasks'],1)
                    self.assertEqual(partial['status'],'incomplete')
            with patch.object(runner,'checked_cache',return_value=self.payload):
                complete=runner.consolidate(out,plan,'synthetic-v9',self.data,self.cfg,lookup,parent_status())
            self.assertEqual(complete['status'],'all_requested_stages_complete')
            self.assertEqual(complete['completed_tasks'],270)
            self.assertEqual(complete['extension_completed_tasks'],15)
            self.assertEqual(complete['stages']['sopls_perblock']['completed_tasks'],15)
            self.assertEqual(complete['combined_v8_and_extension_prediction_rows'],376134+15*198)

    def test_failure_status_survives_a_corrupt_cache_without_claiming_completion(self):
        plan=[self.task]
        lookup={(0,0):self.split}
        with tempfile.TemporaryDirectory(prefix='_synthetic_v9_failure_',dir=str(ROOT)) as folder:
            out=Path(folder)
            (out/'tasks').mkdir()
            path=out/'tasks'/runner.task_name(self.task)
            runner.base.atomic_json(path,dict(signature='wrong',payload_sha256='wrong',payload=self.payload))
            failure=dict(message='Synthetic cache is invalid',task=self.task)
            result=runner.consolidate(out,plan,'synthetic-v9',self.data,self.cfg,lookup,parent_status(),failure=failure)
            self.assertEqual(result['status'],'failed')
            self.assertEqual(result['extension_completed_tasks'],0)
            self.assertEqual(result['completed_tasks'],255)
            self.assertEqual(result['extension_status'],'failed')

    def test_first_task_failure_stops_the_run_and_writes_failure_status(self):
        data=synthetic_dataset(139)
        environment={'pip_freeze':None}
        context={'parent_evidence':{'parent_status':parent_status()}}
        with tempfile.TemporaryDirectory(prefix='_synthetic_v9_abort_',dir=str(ROOT)) as folder:
            with patch.object(runner.base,'load_dataset',return_value=data),patch.object(runner.base,'historical_splits',return_value=splits()),patch.object(runner,'execution_context',return_value=('synthetic-v9',context,environment)),patch.object(runner,'evaluate',side_effect=RuntimeError('Synthetic declared candidate failed')) as failing:
                with self.assertRaisesRegex(RuntimeError,'Synthetic declared candidate failed'):
                    runner.main(['--out',folder])
            self.assertEqual(failing.call_count,1)
            report=json.loads((Path(folder)/'COMPLETENESS.json').read_text())
            self.assertEqual(report['status'],'failed')
            self.assertEqual(report['extension_completed_tasks'],0)
            self.assertEqual(report['completed_tasks'],255)
            self.assertTrue((Path(folder)/'failure.json').exists())
            self.assertEqual(len(list((Path(folder)/'tasks').glob('*.json.gz'))),0)

    def test_original_source_and_parent_run_folders_are_protected(self):
        for name in ('executed','v8','v9','results','data','figures','provenance','configs','docs','tools','results/robustness'):
            with self.subTest(folder=name),self.assertRaisesRegex(ValueError,'immutable'):
                runner.main(['--stage','prepare','--out',str(ROOT/name/'synthetic_output')])

if __name__=='__main__':
    unittest.main()
