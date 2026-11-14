"""Synthetic reuse checks for fixed identities, regression records and payloads."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

RUN=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RUN/'code'))
import state_reuse as reuse
import inference_runner as runner
from state_store import atomic_json,file_sha,member_digest,read_json


def row(identity):
    return dict(global_identity=identity,source_payload_sha256=hashlib.sha256(identity.encode()).hexdigest(),
                real_view_boxes=[[0,0,256,256]],label_stratum='normal')


class TestReuse(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=None)
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.group=dict(reference=[row('fixture:fit')],calibration=[row('fixture:cal:'+str(i)) for i in range(64)],
                        query=[row('fixture:query:0'),row('fixture:query:1')])
        self.config=dict(category='candle',profile_id='M_W_U',budget_id='B',cohort_id='visa12',
                         config_id='fixture_reuse',state_sha256='fixture_state',synthetic_fixture=True)
        self.config['membership_digest']=member_digest(self.group)
        self.data={}
        for role,idkey in [('calibration','normal_ids'),('query','query_ids')]:
            n=len(self.group[role]);a=np.zeros((n,1,256,256),np.float32)
            a[:,0,0,0]=np.arange(n,dtype=np.float32)/100
            scores=np.arange(n,dtype=np.float64)/100
            self.data[role]=dict(maps256=a,object_score=scores,raw32=np.zeros((n,1,32,32),np.float32),
                                **{idkey:np.array([r['global_identity'] for r in self.group[role]])})
        self.calpath=self.root/'historical_cal.npz';np.savez(self.calpath,**self.data['calibration'])
        self.querypath=self.root/'historical_query.npz';np.savez(self.querypath,**self.data['query'])
        self.reg=self.root/'reg';self.reg.mkdir();records=[]
        for i,member in enumerate(self.group['calibration'][:3]):
            p=self.reg/f'normal_{i:03d}.npz'
            np.savez(p,maps256=self.data['calibration']['maps256'][i],object_score=self.data['calibration']['object_score'][i])
            records.append(dict(category='candle',profile_id='M_W_U',budget_id='B',unit_id=member['global_identity'],
                state_sha256='fixture_state',status='PASSED',atol=1e-6,rtol=1e-5,GT_reads=0,
                source_kind='RAW_RGB_RECOMPUTED_NORMAL_TECHNICAL_REGRESSION',mask_equal='NOT_APPLICABLE_UNMASKED',
                actual_encoder_calls=4,actual_PCA_calls=0,prediction_sha256=file_sha(p),
                baseline_path=str(self.calpath),baseline_sha256=file_sha(self.calpath)))
        self.receipt=self.reg/'REGRESSION.json'
        atomic_json(self.receipt,dict(status='PASSED',rows=records,GT_reads=0))
    def test_strict_supported_original_profiles_only(self):
        self.assertEqual(reuse.legacy_profile(self.config),'within_M05')
        with self.assertRaises(ValueError):reuse.legacy_profile(dict(self.config,budget_id='half_B'))
        with self.assertRaises(ValueError):reuse.legacy_profile(dict(self.config,profile_id='I_W_U'))
        with self.assertRaises(ValueError):reuse.legacy_profile(dict(self.config,profile_id='NativeNoRot_full',budget_id='full'))
        self.assertEqual(reuse.legacy_profile(dict(self.config,category='paddy',profile_id='I_W_U')),'within_I01')
    def test_full_order_and_count_required(self):
        reuse._validate_arrays(self.data['calibration'],'calibration',self.group)
        changed=dict(self.data['calibration']);changed['normal_ids']=changed['normal_ids'][::-1]
        with self.assertRaises(ValueError):reuse._validate_arrays(changed,'calibration',self.group)
        changed=dict(self.data['calibration']);changed['maps256']=changed['maps256'][:3]
        with self.assertRaises(ValueError):reuse._validate_arrays(changed,'calibration',self.group)
    def test_nonfinite_and_dtype_refused(self):
        changed=dict(self.data['query']);changed['maps256']=changed['maps256'].astype(np.float64)
        with self.assertRaises(ValueError):reuse._validate_arrays(changed,'query',self.group)
        changed=dict(self.data['query']);changed['object_score']=np.array([1.,np.nan])
        with self.assertRaises(ValueError):reuse._validate_arrays(changed,'query',self.group)
    def test_specific_real_receipt_and_actual_payload_recomputed(self):
        result=reuse._verify_regression(self.config,self.group,'fixture_state',self.receipt,self.data['calibration'],self.calpath)
        self.assertEqual(result['sha256'],file_sha(self.receipt))
        with self.assertRaises(ValueError):
            reuse._verify_regression(self.config,self.group,'another_state',self.receipt,self.data['calibration'],self.calpath)
        doc=read_json(self.receipt);doc['rows'][0]['rtol']=1e-2;atomic_json(self.receipt,doc,replace=True)
        with self.assertRaises(ValueError):
            reuse._verify_regression(self.config,self.group,'fixture_state',self.receipt,self.data['calibration'],self.calpath)
    def test_failed_or_other_profile_receipt_refused(self):
        doc=read_json(self.receipt);doc['rows'][1]['status']='FAILED';atomic_json(self.receipt,doc,replace=True)
        with self.assertRaises(ValueError):
            reuse._verify_regression(self.config,self.group,'fixture_state',self.receipt,self.data['calibration'],self.calpath)
    def test_payload_hash_mutation_refused(self):
        with (self.reg/'normal_000.npz').open('ab') as f:f.write(b'changed')
        with self.assertRaises(ValueError):
            reuse._verify_regression(self.config,self.group,'fixture_state',self.receipt,self.data['calibration'],self.calpath)
    def test_complete_unit_materialization_and_resume_exact_bytes(self):
        descriptor=dict(roles={role:reuse.pin(path) for role,path in [('calibration',self.calpath),('query',self.querypath)]},
                        normal_regression_receipt=reuse.pin(self.receipt),fixture=True)
        config=dict(self.config,normal_regression_receipt_sha256=file_sha(self.receipt))
        out=self.root/'fixture_predictions'
        seal=reuse._materialize(config,self.group,descriptor,out)
        self.assertEqual(seal['source_kind'],'HISTORICAL_REUSED_VERIFIED')
        self.assertTrue(seal['synthetic_fixture'])
        self.assertEqual(len(seal['records']),66)
        with np.load(out/'00064.npz') as copied:
            self.assertTrue(np.array_equal(copied['maps256'],self.data['query']['maps256'][0]))
            self.assertEqual(float(copied['object_score']),float(self.data['query']['object_score'][0]))
        second=reuse._materialize(config,self.group,descriptor,out)
        self.assertEqual(seal,second)
        with (out/'00064.npz').open('ab') as f:f.write(b'changed')
        with self.assertRaises(ValueError):reuse._materialize(config,self.group,descriptor,out)
    def test_production_freeze_checked_before_source_access(self):
        with patch.object(runner,'validate_runtime',side_effect=RuntimeError('no verified fixed protocol')):
            with self.assertRaisesRegex(RuntimeError,'no verified fixed protocol'):
                reuse.reuse_existing(self.config,{'run_root':str(self.root)})
    def test_materialized_seal_connects_to_real_evaluator_loader(self):
        import evaluation as evaluation
        descriptor=dict(roles={role:reuse.pin(path) for role,path in [('calibration',self.calpath),('query',self.querypath)]},
                        normal_regression_receipt=reuse.pin(self.receipt),fixture=True)
        config=dict(self.config,normal_regression_receipt_sha256=file_sha(self.receipt))
        out=self.root/'fixture_evaluator_connection'
        reuse._materialize(config,self.group,descriptor,out)
        seal,loaded=evaluation.validate_prediction_complete(out,self.group,config)
        self.assertEqual(len(loaded),66)
        self.assertEqual(seal['normal_regression_receipt_sha256'],file_sha(self.receipt))
        with self.assertRaises(ValueError):
            evaluation.validate_prediction_complete(out,self.group,dict(config,normal_regression_receipt_sha256='wrong_receipt'))
        with self.assertRaises(ValueError):
            evaluation.evaluate_predictions(out,self.group,config,{}, {},self.root/'forbidden_science',synthetic_fixture=True)


if __name__=='__main__':unittest.main(verbosity=2)
