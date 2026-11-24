from pathlib import Path
import copy
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run
from summarize import summarize


def fixture():
    rng=np.random.RandomState(20261102)
    n=36
    latent=rng.normal(size=(n,3))
    X={m:latent.dot(rng.normal(size=(3,9)))+rng.normal(scale=0.1,size=(n,9)) for m in ('uv','raman','nir')}
    Y=latent.dot(rng.normal(size=(3,11)))+rng.normal(scale=0.1,size=(n,11))+10
    data=SimpleNamespace(X=X,Y=pd.DataFrame(Y,columns=list(run.TARGETS)),ids=np.array(['synthetic_%02d'%i for i in range(n)]),axes={m:np.cumsum(np.linspace(1,2,9)) for m in X})
    cfg=dict(profile='quick',expected_bottles=n,seed=517,repeats=1,outer_folds=2,inner_folds=2,
        learning_sizes=[12],bootstrap_refits=1,run_controls=True,control_permutations=1,control_algorithms=['ridge'],
        algorithms=['pls','ridge','krr','sopls'],panel_tolerances=[0,0.02,0.05,0.10],
        preprocessing_options=dict(uv=['raw'],raman=['raw','snv','baseline_snv'],nir=['raw','snv','sg1']),
        preprocessing_settings=dict(baseline_iterations=3,sg_window=7),
        grid=dict(pls_components=[2],ridge_alphas=[0.1],krr_alphas=[0.1],krr_gammas=[0.3],sopls_components=[1]),
        legacy_grid=dict(snv=[False,True],pls_components=[2],ridge_alphas=[0.1],krr_alphas=[0.1],krr_gammas=[0.3]),
        uncertainty=dict(pca_components=2,ridge_alpha=10),calibration_fraction=0.3,interval_alpha=0.2,
        summary_bootstrap_draws=10,summary_make_figures=False)
    return data,cfg


class RunnerChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data,cls.cfg=fixture()
        cls.splits=run.make_splits(cls.data.ids,cls.cfg)
        cls.plan=run.task_plan(cls.cfg,cls.splits)

    def test_nested_training_sizes_and_roles(self):
        cfg=copy.deepcopy(self.cfg)
        cfg['learning_sizes']=[8,12]
        plan=run.task_plan(cfg,self.splits)
        group=[t for t in plan if t['stage']=='learning' and t['fold']==0]
        group.sort(key=lambda t:int(t['training_size']))
        small=run.indices(cfg,self.splits[0],group[0])[0]
        large=run.indices(cfg,self.splits[0],group[1])[0]
        self.assertTrue(set(small).issubset(large))
        for task in self.plan:
            split=self.splits[task['fold']]
            fit,cal,test,unused=run.indices(self.cfg,split,task)
            self.assertFalse(set(fit)&set(test))
            self.assertFalse(set(fit)&set(cal))
            self.assertFalse(set(cal)&set(test))

    def test_duplicate_outer_bottles_rejected(self):
        with self.assertRaises(ValueError):
            run.make_splits(np.array(['one']*12),self.cfg)

    def test_plan_and_cache_name_identity(self):
        self.assertEqual(len(self.plan),16)
        names=[run.task_name(t) for t in self.plan]
        self.assertEqual(len(names),len(set(names)))
        self.assertTrue(all(n.endswith('.json.gz') for n in names))

    def test_cache_signature_payload_and_task_checked(self):
        task=self.plan[0]
        payload=dict(task,extra='synthetic')
        with tempfile.TemporaryDirectory(prefix='_synthetic_',dir=str(ROOT)) as td:
            path=Path(td)/run.task_name(task)
            run.dump(path,dict(signature='synthetic',payload=payload,payload_sha256=run.payload_hash(payload)))
            self.assertEqual(run.read_cache(path,'synthetic',task),payload)
            with self.assertRaises(RuntimeError):
                run.read_cache(path,'different',task)
            wrong=dict(task,fold=9)
            with self.assertRaises(RuntimeError):
                run.read_cache(path,'synthetic',wrong)
            saved=run.read_json(path)
            saved['payload']['extra']='corrupted'
            run.dump(path,saved)
            with self.assertRaises(RuntimeError):
                run.read_cache(path,'synthetic',task)

    def test_lock_excludes_second_writer(self):
        with tempfile.TemporaryDirectory(prefix='_synthetic_',dir=str(ROOT)) as td:
            with run.Lock(td):
                with self.assertRaises(RuntimeError):
                    with run.Lock(td):
                        pass

    def minimal_payload(self):
        cfg=copy.deepcopy(self.cfg)
        cfg['algorithms']=['ridge']
        cfg['panel_tolerances']=[0]
        task=self.plan[0]
        split=self.splits[0]
        fit,cal,test,unused=run.indices(cfg,split,task)
        sd=self.data.Y.iloc[fit].std(ddof=1).to_numpy()
        outer_sd=self.data.Y.iloc[split['tr']].std(ddof=1).to_numpy()
        rows=[]
        for model in run.engine.model_names(cfg):
            for i in test:
                for j,target in enumerate(self.data.Y.columns):
                    truth=float(self.data.Y.iloc[i,j])
                    rows.append(dict(task,model=model,sample_id=str(self.data.ids[i]),target=target,y_true=truth,y_pred=truth+0.1,train_sd=float(sd[j]),outer_train_sd=float(outer_sd[j]),n_train=len(fit),n_unique_train=len(set(self.data.ids[fit])),n_cal=0))
        return cfg,task,split,dict(task,fit_ids=self.data.ids[fit].tolist(),cal_ids=[],test_ids=self.data.ids[test].tolist(),predictions=rows,intervals=[])

    def test_semantic_cache_rejects_training_bottle_in_test_rows(self):
        cfg,task,split,payload=self.minimal_payload()
        run.validate_payload(payload,self.data,cfg,split,task)
        identity=payload['fit_ids'][0]
        i=np.where(self.data.ids==identity)[0][0]
        target=payload['predictions'][0]['target']
        payload['predictions'][0]['sample_id']=identity
        payload['predictions'][0]['y_true']=float(self.data.Y.loc[i,target])
        with self.assertRaises(RuntimeError):
            run.validate_payload(payload,self.data,cfg,split,task)

    def test_semantic_cache_rejects_row_context_and_budget(self):
        cfg,task,split,payload=self.minimal_payload()
        payload['predictions'][0]['condition']='different'
        with self.assertRaises(RuntimeError):
            run.validate_payload(payload,self.data,cfg,split,task)
        payload['predictions'][0]['condition']=task['condition']
        payload['predictions'][0]['n_train']+=1
        with self.assertRaises(RuntimeError):
            run.validate_payload(payload,self.data,cfg,split,task)

    def test_end_to_end_all_stages(self):
        with tempfile.TemporaryDirectory(prefix='_synthetic_',dir=str(ROOT)) as td:
            out=Path(td)
            for task in self.plan:
                split=self.splits[task['fold']]
                payload=run.run_task(self.data,self.cfg,split,task)
                run.validate_payload(payload,self.data,self.cfg,split,task)
                self.assertEqual(set(row['target'] for row in payload['predictions']),set(run.TARGETS))
                if task['stage']=='reliability':
                    self.assertEqual(len(payload['intervals']),4*len(payload['predictions']))
                    for model,acquisition in payload['diagnostics']['acquisition_by_model'].items():
                        expected=acquisition['modality_union']
                        actual=payload['uncertainty']['models'][model]['available_modalities']
                        self.assertEqual(expected,actual)
                    truth=np.array(payload['calibration_truth'])
                    prediction=np.array(payload['calibration_predictions']['ridge__panel_t005'])
                    sd=self.data.Y.iloc[run.indices(self.cfg,split,task)[0]].std(ddof=1).to_numpy()
                    scores=(abs(truth-prediction)/sd).max(axis=1)
                    rank=int(np.ceil((len(scores)+1)*(1-self.cfg['interval_alpha'])))
                    quantile=np.sort(scores)[rank-1] if rank<=len(scores) else np.inf
                    rows=[r for r in payload['intervals'] if r['model']=='ridge__panel_t005' and r['interval_method']=='constant_joint']
                    np.testing.assert_allclose([r['width']/(2*r['fit_sd']) for r in rows],quantile)
                path=out/'tasks'/run.task_name(task)
                run.dump(path,dict(signature='synthetic',payload=payload,payload_sha256=run.payload_hash(payload)))
            completion=run.consolidate(out,self.cfg,self.splits,self.plan,'synthetic',self.data)
            self.assertTrue(all(r['complete'] for r in completion.values()))
            report=summarize(out,self.cfg)
            self.assertEqual(set(report['available_stages']),set(run.STAGES))
            self.assertEqual(report['execution_status'],'all_requested_stages_complete')
            self.assertTrue((out/'derived'/'report.md').exists() or (out/'derived'/'all_11_targets_report.md').exists())
            cache=out/'tasks'/run.task_name(self.plan[0])
            saved=run.read_json(cache)
            saved['payload']['predictions'][0]['y_true']+=1
            run.dump(cache,dict(signature='synthetic',payload=saved['payload'],payload_sha256=run.payload_hash(saved['payload'])))
            with self.assertRaises(RuntimeError):
                run.consolidate(out,self.cfg,self.splits,self.plan,'synthetic',self.data)


if __name__=='__main__':
    unittest.main()
