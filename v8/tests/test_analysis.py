import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from v8.analysis import TARGETS, holm_adjust, corrected_cv, bootstrap_draws, conditional_pair, normalize_predictions, validate_complete_pair, analyze_run


class AnalysisChecks(unittest.TestCase):
    def test_holm_preserves_the_planned_family_and_monotonicity(self):
        np.testing.assert_allclose(holm_adjust([.01,.04,.03]),[.03,.06,.06])
        result=holm_adjust([.01,np.nan,.04])
        np.testing.assert_allclose(result[[0,2]],[.03,.08])
        self.assertTrue(np.isnan(result[1]))
        with self.assertRaises(ValueError):
            holm_adjust([-.1,.2])

    def test_shared_bottle_draw_preserves_all_eleven_target_relationships(self):
        base=np.array([1.,4.,9.,16.])[:,None]
        scale=np.arange(1,12,dtype=float)[None,:]
        before=base*scale
        after=before*.81
        indices=bootstrap_draws(4,200,83)
        result=conditional_pair(before,after,before,after,indices,chunk_size=17)
        np.testing.assert_allclose(result['bootstrap_target_gains'],10.,rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(result['target_gain_bounds'],10.,rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(result['bootstrap_target_gains'][:,[0]],result['bootstrap_target_gains'][:,[10]],rtol=1e-13,atol=1e-13)
        self.assertLess(result['macro_mse_difference_after_minus_before'],0)

    def test_self_contrast_is_exactly_zero(self):
        before=np.arange(1,56,dtype=float).reshape(5,11)
        result=conditional_pair(before,before,before,before,bootstrap_draws(5,99,38))
        np.testing.assert_array_equal(result['target_gain_pct'],np.zeros(11))
        np.testing.assert_array_equal(result['target_gain_bounds'],np.zeros((2,11)))
        np.testing.assert_array_equal(result['macro_mse_difference_bounds'],np.zeros(2))
        corrected=corrected_cv(np.zeros(15),np.full(15,.25))
        self.assertEqual(corrected['p_uncorrected'],1.)
        self.assertEqual(corrected['corrected_se'],0.)

    def test_cross_target_bootstrap_uses_the_same_bottle_draw(self):
        scales=np.arange(1,12,dtype=float)[None,:]
        before=np.array([1.,4.,9.,16.])[:,None]*scales
        after=np.array([.5,6.,7.,20.])[:,None]*scales
        result=conditional_pair(before,after,before,after,bootstrap_draws(4,150,62),chunk_size=11)
        self.assertGreater(result['bootstrap_target_gains'][:,0].std(),.1)
        np.testing.assert_allclose(result['bootstrap_target_gains'][:,0],result['bootstrap_target_gains'][:,10],rtol=1e-12,atol=1e-12)

    def test_corrected_cv_does_not_use_independent_fold_standard_error(self):
        losses=np.linspace(-.1,.2,15)
        result=corrected_cv(losses,np.full(15,.25))
        naive=losses.std(ddof=1)/np.sqrt(15)
        self.assertAlmostEqual(result['corrected_se']/naive,np.sqrt(4.75))
        self.assertIn('Approximate',result['inference_scope'])

    def test_missing_training_roles_and_unpaired_bottles_are_rejected(self):
        with self.assertRaises(ValueError):
            normalize_predictions(pd.DataFrame({'model':['x']}))
        rows=[]
        for model in ('a','b'):
            for bottle in ('one','two'):
                for target in TARGETS:
                    rows.append(dict(condition='test',training_size='full',model=model,repeat=0,fold=0,
                        sample_id=bottle,target=target,y_true=1.,y_pred=1.1,train_sd=1.,outer_train_sd=1.,n_train=3,n_unique_train=3))
        frame=normalize_predictions(pd.DataFrame(rows))
        a=frame[frame.model.eq('a')]
        b=frame[frame.model.eq('b')]
        validate_complete_pair(a,b,['one','two'],require_full=False)
        with self.assertRaises(ValueError):
            validate_complete_pair(a,b[b.sample_id.eq('one')],['one','two'],require_full=False)
        with self.assertRaises(ValueError):
            validate_complete_pair(a,b,['one','two'],require_full=True)

    def test_run_analysis_separates_complete_conditional_and_learning_contexts(self):
        with tempfile.TemporaryDirectory(prefix='v8_analysis_') as directory:
            root=Path(directory)
            data=root/'executed/data'
            data.mkdir(parents=True)
            ids=['synthetic_%03d'%i for i in range(139)]
            pd.DataFrame({'sample_id':ids}).to_csv(data/'chemistry.csv',index=False)
            run=root/'run'
            run.mkdir()
            rows=[]
            for condition,budget in [('nir_weight','full'),('nir_weight_learning','40')]:
                for repeat in range(3):
                    order=np.random.RandomState(17+repeat).permutation(139)
                    for fold,test in enumerate(np.array_split(order,5)):
                        n_train=139-len(test) if budget=='full' else 40
                        for model,factor,weight in [('krr__weight_zero',1.,0.),('krr__weight_selected',.8,.25)]:
                            for i in test:
                                for j,target in enumerate(TARGETS):
                                    truth=(i+1.)*(j+1.)/100.
                                    error=((i%7)+1.)*(j+1.)/1000.
                                    rows.append(dict(condition=condition,training_size=budget,model=model,repeat=repeat,
                                        fold=fold,budget_refit=-1,sample_id=ids[i],target=target,y_true=truth,
                                        y_pred=truth+factor*error,train_sd=j+1.,outer_train_sd=j+1.,
                                        n_train=n_train,n_unique_train=n_train,effective_nir_weight=weight))
            pd.DataFrame(rows).to_csv(run/'predictions.csv.gz',index=False,compression='gzip')
            registry=analyze_run(root,run,dict(statistical_bootstrap_draws=40,statistics_seed=17))
            self.assertEqual(registry['n_independent_bottles'],139)
            intervals=pd.read_csv(run/'derived/target_conditional_intervals.csv')
            self.assertEqual(len(intervals),11)
            self.assertEqual(set(intervals.condition),{'nir_weight'})
            self.assertTrue(intervals.n_prediction_visits.eq(417).all())
            descriptive=pd.read_csv(run/'derived/paired_target_effects.csv')
            self.assertEqual(len(descriptive),22)
            panels=pd.read_csv(run/'derived/panel_by_repeat_and_refit.csv')
            full=panels[panels.condition.eq('nir_weight')]
            self.assertTrue(full.n_train_min.eq(111).all())
            self.assertTrue(full.n_train_max.eq(112).all())
            self.assertTrue(full.n_unique_train_min.eq(111).all())
            self.assertTrue(full.n_unique_train_max.eq(112).all())
            statuses=pd.read_csv(run/'derived/conditional_context_status.csv')
            self.assertEqual(set(statuses.status),{'complete_paired_conditional_analysis','learning_or_refit_descriptive_only'})
            sensitivity=pd.read_csv(run/'derived/target_corrected_cv_sensitivity.csv')
            self.assertTrue(sensitivity.planned_family_size.eq(11).all())
            self.assertTrue((run/'analysis_registry.json').exists())


if __name__=='__main__':
    unittest.main()
