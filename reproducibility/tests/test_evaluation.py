"""Synthetic integration/failure tests only. No original image/GT/model opened."""
from __future__ import annotations
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('evaluation',ROOT/'code/evaluation.py')
ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)


class EvaluationIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='evaluation_fixture_',dir=None)
        cls.fixture=Path(cls.temp.name)/'fixture_e2e'
        cls.receipt=ev.run_synthetic_fixture(cls.fixture)
        cls.pred=cls.fixture/'M_W_U_predictions'
        cls.seal=ev.read(cls.pred/'COMPLETE.json')
        cls.config={k:cls.seal[k] for k in ['config_id','category','profile_id','budget_id','cohort_id','state_sha256','membership_digest']}
        def unit(identity,label):
            return dict(global_identity=identity,label_stratum=label,real_view_boxes=[[0,0,256,256],[256,0,512,256]],composite_wh=[512,256])
        cls.group=dict(reference=[unit('fixture:fit:0','normal')],
            calibration=[unit(f'fixture:cal:{i}','normal') for i in range(64)],
            query=[unit('fixture:q:normal','normal'),unit('fixture:q:bad','anomaly')])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def copy_predictions(self):
        import shutil
        target=Path(self.temp.name)/('pred_'+ev.uuid.uuid4().hex)
        shutil.copytree(self.pred,target)
        return target

    def test_fixture_complete_and_fixed_statistics(self):
        self.assertEqual(self.receipt['status'],'SYNTHETIC_E2E_COMPLETE_NOT_SCIENCE')
        self.assertFalse(self.receipt['original_GT_opened'])
        self.assertEqual(self.receipt['bootstrap_n'],10000)
        self.assertEqual(self.receipt['bootstrap_seed'],20261022)
        c=ev.read(self.fixture/'fixture_statistics/COMPLETE.json')
        self.assertEqual(c['estimates_n'],1)

    def test_manifest_exact_order_and_source_fields(self):
        seal,data=ev.validate_prediction_complete(self.pred,self.group,self.config)
        self.assertEqual(len(data),66)
        self.assertEqual(data[0]['maps256'].shape,(2,256,256))
        self.assertTrue(all(r['role']=='calibration' for r in seal['records'][:64]))

    def test_missing_complete_refused_before_GT(self):
        target=Path(self.temp.name)/'empty_predictions';target.mkdir(exist_ok=True)
        with self.assertRaisesRegex(ValueError,'COMPLETE.json'):
            ev.validate_prediction_complete(target,self.group,self.config)

    def test_partial_inventory_refused(self):
        target=self.copy_predictions();seal=ev.read(target/'COMPLETE.json');seal['records'].pop()
        (target/'COMPLETE.json').write_bytes(ev.json_bytes(seal))
        with self.assertRaisesRegex(ValueError,'Incomplete calibration/query'):
            ev.validate_prediction_complete(target,self.group,self.config)

    def test_wrong_state_refused(self):
        config=dict(self.config,state_sha256='f'*64)
        with self.assertRaisesRegex(ValueError,'state_sha256'):
            ev.validate_prediction_complete(self.pred,self.group,config)

    def test_wrong_order_refused(self):
        target=self.copy_predictions();seal=ev.read(target/'COMPLETE.json')
        seal['records'][0],seal['records'][1]=seal['records'][1],seal['records'][0]
        (target/'COMPLETE.json').write_bytes(ev.json_bytes(seal))
        with self.assertRaisesRegex(ValueError,'member order'):
            ev.validate_prediction_complete(target,self.group,self.config)

    def test_payload_tamper_refused(self):
        target=self.copy_predictions();p=target/self.seal['records'][0]['file']
        with p.open('ab') as f:f.write(b'tamper')
        with self.assertRaisesRegex(ValueError,'file hash'):
            ev.validate_prediction_complete(target,self.group,self.config)

    def test_discrete_slot_count_refused(self):
        target=self.copy_predictions();seal=ev.read(target/'COMPLETE.json');seal['records'][0]['actual_views']=1
        (target/'COMPLETE.json').write_bytes(ev.json_bytes(seal))
        with self.assertRaisesRegex(ValueError,'acquisition-slot'):
            ev.validate_prediction_complete(target,self.group,self.config)

    def test_role_leakage_refused(self):
        group=dict(self.group,reference=[self.group['calibration'][0]])
        with self.assertRaisesRegex(ValueError,'Role leakage'):
            ev.validate_prediction_complete(self.pred,group,self.config)

    def test_calibration_count_cannot_be_reduced(self):
        group=dict(self.group,calibration=self.group['calibration'][:-1])
        with self.assertRaisesRegex(ValueError,'exactly 64'):
            ev.validate_prediction_complete(self.pred,group,self.config)

    def test_unknown_profile_refused(self):
        config=dict(self.config,profile_id='within_I_typo')
        with self.assertRaisesRegex(ValueError,'Unregistered'):
            ev.validate_prediction_complete(self.pred,self.group,config)

    def test_GT_not_opened_without_gate(self):
        with patch.object(ev,'_read_gt',side_effect=AssertionError('GT must not open')) as gt:
            with self.assertRaisesRegex(ValueError,'fixed-protocol record'):
                ev.evaluate_predictions(self.pred,self.group,self.config,ev.SOURCE_DEFAULTS,
                    dict(kind='directory',path='DO_NOT_OPEN'),Path(self.temp.name)/'blocked_output')
            gt.assert_not_called()

    def test_production_has_no_synthetic_fallback(self):
        with self.assertRaisesRegex(ValueError,'no synthetic fallback'):
            ev.evaluate_predictions(self.pred,self.group,self.config,ev.SOURCE_DEFAULTS,{},
                Path(self.temp.name)/'refused_output',synthetic_fixture=True)

    def test_normal_AP_is_blank_not_zero(self):
        units=ev._csv_rows(self.fixture/'M_W_U_evaluation/PER_UNIT.csv')
        normal=units[0]
        self.assertEqual(normal['object_pixel_AP'],'')
        self.assertEqual(normal['pixel_AP_eligible'],'False')
        self.assertEqual(normal['AP_eligibility_reason'],'normal_unit')

    def test_whole_unit_metrics_join_views_not_mean_view_AP(self):
        stats=ev._pinned_module(ev.SOURCE_DEFAULTS,'statistics_core')
        seal,loaded=ev.validate_prediction_complete(self.pred,self.group,self.config)
        for x in loaded:x['maps256'].fill(0)
        loaded[-1]['maps256'][0,0,0]=np.float32(.8) # one GT positive
        loaded[-1]['maps256'][1,0,0]=np.float32(.9) # negative in another view ranks first
        gt=np.zeros((2,2,256,256),dtype=bool);gt[1,0,0,0]=True
        _,units=ev._compute_evaluation(self.group,self.config,seal,loaded,gt,np.array([False,True]),stats.binary_metrics)
        self.assertEqual(units[1]['object_pixel_AP'],.5)
        self.assertEqual(units[1]['actual_views'],2)

    def test_empty_positive_anomaly_AP_is_missing(self):
        stats=ev._pinned_module(ev.SOURCE_DEFAULTS,'statistics_core')
        seal,loaded=ev.validate_prediction_complete(self.pred,self.group,self.config)
        gt=np.zeros((2,2,256,256),dtype=bool)
        summary,units=ev._compute_evaluation(self.group,self.config,seal,loaded,gt,np.array([False,True]),stats.binary_metrics)
        self.assertIsNone(units[1]['object_pixel_AP'])
        self.assertEqual(units[1]['AP_eligibility_reason'],'abnormal_zero_positive_GT')
        self.assertIsNone(summary['mean_object_pixel_AP'])

    def test_missing_fixed_macro_category_refused(self):
        requirement=dict(cohort_id='visa12',stratum='all12',category_or_macro='all12',left_profile='M_W_U',
                         right_profile='I_W_U',left_budget='B',right_budget='B')
        with self.assertRaisesRegex(ValueError,'Incomplete fixed category'):
            ev._bootstrap_impl([self.fixture/'M_W_U_evaluation',self.fixture/'I_W_U_evaluation'],
                [requirement],ev.SOURCE_DEFAULTS,Path(self.temp.name)/'fixture_missing_macro',True)

    def test_original_bootstrap_reproducible_and_no_view_resampling(self):
        panel=ev._pinned_module(ev.SOURCE_DEFAULTS,'panel_statistics')
        _,_,a=ev.validate_evaluation_complete(self.fixture/'M_W_U_evaluation')
        _,_,b=ev.validate_evaluation_complete(self.fixture/'I_W_U_evaluation')
        selected=[dict(x,method='LEFT') for x in a]+[dict(x,method='RIGHT') for x in b]
        x=panel.contrasts(selected,{'candle':['candle']},[('LEFT','RIGHT')],'visa12')
        y=panel.contrasts(selected,{'candle':['candle']},[('LEFT','RIGHT')],'visa12')
        self.assertEqual(x,y)
        self.assertTrue(all(r['bootstrap_replicates']==10000 and r['bootstrap_seed']==20261022 for r in x))
        ap=next(r for r in x if r['endpoint']=='object_pixel_AP')
        self.assertEqual(ap['eligible_units'],1)

    def test_source_changed_refused(self):
        sources=dict(ev.SOURCE_DEFAULTS)
        sources['statistics_core']=dict(sources['statistics_core'],sha256='0'*64)
        with self.assertRaisesRegex(ValueError,'SHA changed'):
            ev._pinned_module(sources,'statistics_core')

    def test_atomic_failure_is_not_success(self):
        out=Path(self.temp.name)/'atomic_failure'
        def fail(stage):
            (stage/'uncommitted.txt').write_text('fixture',encoding='utf-8')
            raise ValueError('deliberate fixture failure')
        with self.assertRaisesRegex(ValueError,'deliberate'):
            ev._atomic_directory(out,fail)
        self.assertFalse(out.exists())
        self.assertTrue(list(out.parent.glob('atomic_failure.partial.*')))

    def _synthetic_gt_source(self):
        from PIL import Image
        folder=Path(self.temp.name)/('fixture_gt_'+ev.uuid.uuid4().hex);folder.mkdir()
        x=np.zeros((256,512),dtype=np.uint8);x[0,0]=1;x[0,256]=255
        image=folder/'mask.png';Image.fromarray(x).save(image)
        group={k:list(v) for k,v in self.group.items()}
        group['query']=[dict(v) for v in group['query']]
        group['query'][1].update(gt_member='mask.png',GT_payload_sha256=ev.sha(image))
        return folder,group,image

    def test_original_GT_positive_then_nearest256_on_synthetic_PNG(self):
        folder,group,_=self._synthetic_gt_source()
        gt,bad=ev._read_gt(group,dict(kind='directory',path=str(folder)))
        self.assertEqual(gt.shape,(2,2,256,256))
        self.assertEqual(gt.dtype,bool)
        self.assertEqual(gt[1].sum(),2)
        self.assertTrue(gt[1,0,0,0] and gt[1,1,0,0])
        self.assertEqual(bad.tolist(),[False,True])

    def test_tar_gz_uses_pinned_archive_and_explicit_members(self):
        import tarfile
        folder,group,image=self._synthetic_gt_source()
        archive=folder/'fixture_masks.tar.gz'
        with tarfile.open(archive,'w:gz') as out:out.add(image,arcname='mask.png')
        gt,_=ev._read_gt(group,dict(kind='tar_gz',path=str(archive),sha256=ev.sha(archive)))
        self.assertEqual(gt[1].sum(),2)
        with self.assertRaisesRegex(ValueError,'archive SHA'):
            ev._read_gt(group,dict(kind='tar_gz',path=str(archive),sha256='0'*64))

    def test_directory_GT_payload_changed_refused(self):
        folder,group,image=self._synthetic_gt_source()
        group['query'][1]['GT_payload_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'GT member SHA'):
            ev._read_gt(group,dict(kind='directory',path=str(folder)))


if __name__=='__main__':
    unittest.main(verbosity=2)
