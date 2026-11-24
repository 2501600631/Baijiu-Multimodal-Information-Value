from pathlib import Path
import copy
import gzip
import json
import tempfile
import unittest
import numpy as np
import pandas as pd
from v9.analysis import METHODS, MASKS, PRIMARY, SECONDARY, TARGETS, analyze, compute_comparisons, selected_stage_scores, selection_table
from v8.analysis import holm_adjust

def synthetic_predictions():
    ids=np.array(['b%03d'%value for value in range(139)])
    rng=np.random.RandomState(731)
    truth=rng.normal(size=(139,11))+np.arange(11)[None,:]
    noise=rng.normal(size=(139,11))
    rows=[]
    for repeat in range(3):
        partitions=np.array_split(np.random.RandomState(701+repeat).permutation(139),5)
        for fold,test in enumerate(partitions):
            for method in METHODS:
                for mask in MASKS:
                    multiplier={'pls':1.2,'ridge':1.0,'krr':.8,'sopls':.9,'sopls_perblock':.9}[method]
                    if mask==MASKS[1]:
                        multiplier*=.85 if method=='sopls_perblock' else 1.05
                    for i in test:
                        for j,target in enumerate(TARGETS):
                            error=multiplier*(noise[i,j]+.02*repeat+.01*fold)
                            rows.append(dict(condition='sopls_perblock' if method=='sopls_perblock' else 'enhanced',
                                             training_size='full',budget_refit=-1,bootstrap_refit=-1,
                                             model=method+'__'+mask,repeat=repeat,fold=fold,sample_id=ids[i],target=target,
                                             y_true=truth[i,j],y_pred=truth[i,j]+error,train_sd=1+j*.1,outer_train_sd=1+j*.1,
                                             n_train=139-len(test),n_unique_train=139-len(test)))
    return pd.DataFrame(rows),ids.tolist()

class StructuralAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame,cls.ids=synthetic_predictions()
        cls.cfg=dict(statistical_bootstrap_draws=80,statistics_seed=20261104)
        cls.result=compute_comparisons(cls.frame,cls.ids,cls.cfg,True)

    def test_primary_and_secondary_families_include_every_historical_and_new_method(self):
        target=self.result['target_corrected_cv_joined_family']
        panel=self.result['panel_corrected_cv_joined_family']
        self.assertEqual(len(target[target.family.eq(PRIMARY)]),55)
        self.assertEqual(len(target[target.family.eq(SECONDARY)]),22)
        self.assertEqual(len(panel[panel.family.eq(PRIMARY)]),5)
        self.assertEqual(len(panel[panel.family.eq(SECONDARY)]),2)
        self.assertTrue(target[target.family.eq(PRIMARY)].planned_family_size.eq(55).all())
        self.assertTrue(panel[panel.family.eq(PRIMARY)].planned_family_size.eq(5).all())
        self.assertEqual(set(panel[panel.family.eq(PRIMARY)].method),{'pls','ridge','krr','sopls_uniform','sopls_perblock'})

    def test_holm_is_recomputed_jointly_not_a_separate_one_test_family(self):
        for name in ('target_corrected_cv_joined_family','panel_corrected_cv_joined_family'):
            table=self.result[name]
            for family in (PRIMARY,SECONDARY):
                group=table[table.family.eq(family)]
                np.testing.assert_allclose(group.p_holm,holm_adjust(group.p_uncorrected.to_numpy()),equal_nan=True,atol=0,rtol=0)
        primary=self.result['panel_corrected_cv_joined_family']
        self.assertFalse(primary.planned_family_size.eq(1).any())

    def test_identical_uniform_and_perblock_fixed_mask_has_exact_zero_effect_and_interval(self):
        effects=self.result['panel_conditional_intervals']
        row=effects[effects.family.eq(SECONDARY)&effects.comparison.eq(MASKS[0])].iloc[0]
        self.assertEqual(row.gain_pct,0)
        self.assertEqual(row.conditional_gain_ci_low_pct,0)
        self.assertEqual(row.conditional_gain_ci_high_pct,0)
        cv=self.result['panel_corrected_cv_joined_family']
        tested=cv[cv.family.eq(SECONDARY)&cv.comparison.eq(MASKS[0])].iloc[0]
        self.assertEqual(tested.p_uncorrected,1)
        self.assertEqual(tested.p_holm,1)

    def test_all_eleven_targets_and_raw_error_direction_are_retained(self):
        target=self.result['paired_target_effects']
        self.assertEqual(len(target),77)
        for values,group in target.groupby(['family','method','comparison']):
            self.assertEqual(set(group.target),set(TARGETS))
            self.assertEqual(len(group),11)
        panel=self.result['paired_panel_effects']
        self.assertEqual(len(panel),7)
        old=panel[panel.method.eq('sopls_uniform')].iloc[0]
        new=panel[panel.method.eq('sopls_perblock')].iloc[0]
        self.assertLess(old.gain_pct,0)
        self.assertGreater(new.gain_pct,0)
        self.assertEqual(len(self.result['paired_task_comparison']),105)

    def test_incomplete_new_tasks_have_descriptive_outputs_without_intervals_or_p_values(self):
        frame=self.frame[~self.frame.model.str.startswith('sopls_perblock__')|self.frame.fold.eq(0)].copy()
        result=compute_comparisons(frame,self.ids,self.cfg,False)
        self.assertTrue(result['panel_conditional_intervals'].empty)
        self.assertTrue(result['target_conditional_intervals'].empty)
        self.assertTrue(result['panel_corrected_cv_joined_family'].empty)
        self.assertTrue(result['target_corrected_cv_joined_family'].empty)
        self.assertTrue(result['paired_panel_effects'].analysis_status.eq('partial_descriptive_only').all())
        self.assertTrue(result['comparison_status'].conditional_intervals.eq(False).all())
        self.assertIsNone(result['draw_metadata']['draw_indices_sha256'])

    def test_complete_status_cannot_hide_an_absent_outer_fold(self):
        frame=self.frame[~self.frame.model.str.startswith('sopls_perblock__')|~(self.frame.repeat.eq(0)&self.frame.fold.eq(0))].copy()
        with self.assertRaises(ValueError):
            compute_comparisons(frame,self.ids,self.cfg,True)

    def test_paired_reference_concentration_and_training_scales_are_enforced(self):
        for column in ('y_true','train_sd'):
            changed=self.frame.copy()
            position=changed.index[changed.model.eq('sopls_perblock__'+MASKS[1])][0]
            changed.loc[position,column]+=0.1
            with self.subTest(column=column),self.assertRaises(ValueError):
                compute_comparisons(changed,self.ids,self.cfg,True)

    def test_bootstrap_index_matrix_is_identical_across_targets_and_models(self):
        metadata=self.result['draw_metadata']
        repeated=compute_comparisons(self.frame,self.ids,self.cfg,True)
        self.assertEqual(metadata['draw_indices_sha256'],repeated['draw_metadata']['draw_indices_sha256'])
        pd.testing.assert_frame_equal(self.result['target_conditional_intervals'],repeated['target_conditional_intervals'])
        self.assertTrue(metadata['all_targets_and_models_share_indices'])

    def test_component_counts_follow_order_and_active_counts_are_retained(self):
        model='sopls_perblock__'+MASKS[1]
        selections=[]
        finals=[]
        for j in range(11):
            spec=dict(order=['nir','uv','raman'],components=[1,2,5],components_by_block=[1,2,5],preprocessing={'nir':'raw','uv':'raw','raman':'snv'})
            selections.append(dict(target_index=j,candidate_id=3,path_id=3,spec=spec,inner_mse=.2,
                                   stage_inner_scores=[dict(stage=1,modality='nir',selected_k=1,inner_mse=.4),dict(stage=2,modality='uv',selected_k=2,inner_mse=.3),dict(stage=3,modality='raman',selected_k=5,inner_mse=.2)]))
            finals.append(dict(target_index=j,details=dict(total_components_active=6,blocks=[dict(modality='nir',components_active=1),dict(modality='uv',components_active=1),dict(modality='raman',components_active=4)])))
        payload=dict(task=dict(repeat=0,fold=1),diagnostics=dict(selections={model:selections},final_fits={model:finals}))
        table=selection_table([payload],'new_perblock')
        self.assertEqual(len(table),11)
        self.assertTrue(table.k1.eq(1).all())
        self.assertTrue(table.k2.eq(2).all())
        self.assertTrue(table.k3.eq(5).all())
        self.assertTrue(table.k_uv.eq(2).all())
        self.assertTrue(table.k_raman.eq(5).all())
        self.assertTrue(table.k_nir.eq(1).all())
        self.assertTrue(table.active_total_components.eq(6).all())
        self.assertTrue(table.active_k3.eq(4).all())
        self.assertTrue(table.stage1_inner_mse.eq(.4).all())
        self.assertTrue(table.stage3_inner_mse.eq(.2).all())
        stages=selected_stage_scores([payload])
        self.assertEqual(len(stages),33)
        self.assertEqual(set(stages.target),set(TARGETS))

    def test_complete_analysis_exports_old_new_components_stages_and_joined_families(self):
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix='_analysis_synthetic_',dir=str(root)) as folder:
            folder=Path(folder)
            (folder/'executed/data').mkdir(parents=True)
            pd.DataFrame(dict(sample_id=self.ids)).to_csv(folder/'executed/data/chemistry.csv',index=False)
            old_dir=folder/'results/baseline'
            (old_dir/'benchmark').mkdir(parents=True)
            (old_dir/'tasks').mkdir()
            run=folder/'runs/new'
            (run/'tasks').mkdir(parents=True)
            old=self.frame[~self.frame.model.str.startswith('sopls_perblock__')]
            new=self.frame[self.frame.model.str.startswith('sopls_perblock__')]
            old.to_csv(old_dir/'benchmark/predictions.csv.gz',index=False)
            new.to_csv(run/'predictions.csv.gz',index=False)
            (run/'COMPLETENESS.json').write_text(json.dumps(dict(extension_completed_tasks=15)))
            for repeat in range(3):
                for fold in range(5):
                    for method,destination in [('sopls',old_dir/'tasks'),('sopls_perblock',run/'tasks')]:
                        selections,finals={},{}
                        for mask in MASKS:
                            order=mask.split('+')
                            model=method+'__'+mask
                            selections[model]=[]
                            finals[model]=[]
                            for target in range(11):
                                spec=dict(order=order,components=[2]*len(order) if method=='sopls_perblock' else 2,preprocessing={modality:'raw' for modality in order})
                                item=dict(target_index=target,candidate_id=target,spec=spec,inner_mse=.1)
                                if method=='sopls_perblock':
                                    item['stage_inner_scores']=[dict(stage=stage+1,modality=modality,selected_k=2,inner_mse=.1) for stage,modality in enumerate(order)]
                                selections[model].append(item)
                                finals[model].append(dict(target_index=target,details=dict(total_components_active=2*len(order),blocks=[dict(modality=modality,components_active=2) for modality in order])))
                        payload=dict(task=dict(repeat=repeat,fold=fold),diagnostics=dict(selections=selections,final_fits=finals))
                        with gzip.open(str(destination/('benchmark_r%d_f%d.json.gz'%(repeat,fold))),'wt',encoding='utf-8') as stream:
                            json.dump(dict(payload=payload),stream)
            registry=analyze(folder,run,self.cfg)
            self.assertEqual(registry['status'],'complete_extension_postprocessed')
            self.assertEqual(registry['completed_extension_tasks'],15)
            self.assertEqual(registry['tables']['parameter_selection_by_task_target']['rows'],660)
            self.assertEqual(registry['tables']['selected_stage_scores']['rows'],825)
            self.assertEqual(registry['tables']['target_conditional_intervals']['rows'],77)
            self.assertEqual(registry['tables']['panel_conditional_intervals']['rows'],7)
            self.assertEqual(registry['tables']['paired_task_comparison']['rows'],105)
            self.assertEqual(registry['tables']['uniform_perblock_paired_task_parameters']['rows'],330)
            paired=pd.read_csv(run/'derived/uniform_perblock_paired_task_parameters.csv')
            self.assertEqual(len(paired[['repeat','fold','method']].drop_duplicates()),30)
            self.assertTrue(paired.groupby(['repeat','fold','method']).size().eq(11).all())
            self.assertTrue(paired.ur_k3.isna().all())
            self.assertTrue(paired.urn_k3.eq(2).all())
            self.assertTrue(paired.groupby(['repeat','fold','method']).macro_mse_before.nunique().eq(1).all())
            self.assertFalse(registry['historical_tables_overwritten'])
            self.assertEqual(registry['model_fits_performed'],0)

if __name__=='__main__':
    unittest.main()
