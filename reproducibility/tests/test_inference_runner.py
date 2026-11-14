"""Synthetic structural fixtures only. No actual RGB, pretrained model or GT."""
from __future__ import annotations
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

RUN=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RUN/'code'))
import inference_runner as r
from state_store import atomic_json,digest,file_sha,member_digest,read_json


def row(identity,views=1):
    return dict(global_identity=identity,source_payload_sha256='fixture_'+identity,
                real_view_boxes=[[0,0,1,1]]*views,label_stratum='normal')


def group():
    return dict(reference=[row('fit')],calibration=[row('cal0'),row('cal1'),row('cal2')],
                query=[row('query0'),row('query1',2)])


def config():
    return dict(config_id='synthetic_config',category='fixture',profile_id='M_W_U',
                budget_id='B',cohort_id='SYNTHETIC_FIXTURE',state_sha256='synthetic_state')


class Runtime:
    def __init__(self,fail_at=None):
        self.calls=0;self.fail_at=fail_at
    def predict_views(self,views):
        self.calls+=1
        if self.calls==self.fail_at:raise RuntimeError('intentional fixture interruption')
        return dict(maps256=np.zeros((len(views),256,256),np.float32),object_score=np.array(1.0),
                    encoder_readings_per_view=4,distance_queries_per_anchor=1)


class Backend:
    @staticmethod
    def decode_unit(row,context,verify=True):
        if not verify:raise AssertionError('Scientific fixture resume must verify source')
        return [np.zeros((1,1,3),np.uint8) for _ in row['real_view_boxes']]


class TestRunner(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=None)
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.context=dict(run_root=str(self.root),gpu_safety_checked=False)
    def test_parent_full_plan_and_shared_states(self):
        p=r.plan(run_root=self.root)
        self.assertEqual(p['counts']['requirements'],148)
        self.assertEqual(p['counts']['quality_requirements'],123)
        self.assertEqual(p['counts']['timing_requirements'],25)
        self.assertEqual(p['counts']['unique_quality_configs'],109)
        nc=next(c for c in p['configs'] if c['category']=='candle' and c['profile_id']=='N_W_U')
        ic=next(c for c in p['configs'] if c['category']=='candle' and c['profile_id']=='I_W_U')
        mc=next(c for c in p['configs'] if c['category']=='candle' and c['profile_id']=='M_W_U')
        mf=next(c for c in p['configs'] if c['category']=='candle' and c['profile_id']=='M_W_FC4')
        self.assertEqual(nc['state_id'],ic['state_id'])
        self.assertEqual(mc['state_id'],mf['state_id'])
        self.assertNotEqual(mc['state_id'],nc['state_id'])
    def test_fixed_paddy_budget_not_replicated(self):
        p=r.plan(run_root=self.root)
        c=next(c for c in p['configs'] if c['category']=='paddy' and
               c['profile_id']=='NativeNoRot_pooled_U' and c['budget_id']=='B')
        self.assertEqual(c['k_per_slot'],[7208]*5)
        self.assertEqual(sum(c['k_per_slot']),36040)
    def test_unknown_profile_fails_closed(self):
        with self.assertRaises(ValueError):r.strict_profile('within_I_named_but_raw_fallback')
    def test_missing_runtime_bundle_stops_before_model(self):
        with self.assertRaisesRegex(ValueError, 'raw-rebuild bundle'):
            r.validate_runtime(self.context)
        self.assertFalse(self.context['gpu_safety_checked'])

    def test_atomic_prediction_interruption_resume_and_seal(self):
        out=self.root/'predictions';g=group();cfg=config()
        first=Runtime(fail_at=2)
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            r.predict_with_runtime(cfg,self.context,first,g,out,Backend,source_kind='SYNTHETIC_FIXTURE')
        self.assertFalse((out/'COMPLETE.json').exists())
        self.assertEqual(len(read_json(out/'PROGRESS.json')['records']),1)
        second=Runtime()
        seal=r.predict_with_runtime(cfg,self.context,second,g,out,Backend,source_kind='SYNTHETIC_FIXTURE')
        self.assertEqual(second.calls,4)
        self.assertEqual(seal['source_kind'],'SYNTHETIC_FIXTURE')
        self.assertEqual([x['role'] for x in seal['records']],['calibration']*3+['query']*2)
        self.assertEqual(seal['membership_digest'],member_digest(g))
        third=Runtime()
        r.predict_with_runtime(cfg,self.context,third,g,out,Backend,source_kind='SYNTHETIC_FIXTURE')
        self.assertEqual(third.calls,0)
        with (out/'00000.npz').open('ab') as f:f.write(b'changed fixture')
        with self.assertRaises(ValueError):
            r.predict_with_runtime(cfg,self.context,Runtime(),g,out,Backend,source_kind='SYNTHETIC_FIXTURE')
    def test_resume_rejects_changed_state_and_role_leak(self):
        out=self.root/'pred';g=group()
        with self.assertRaises(RuntimeError):
            r.predict_with_runtime(config(),self.context,Runtime(2),g,out,Backend,source_kind='SYNTHETIC_FIXTURE')
        changed=dict(config(),state_sha256='different')
        with self.assertRaises(ValueError):
            r.predict_with_runtime(changed,self.context,Runtime(),g,out,Backend,source_kind='SYNTHETIC_FIXTURE')
        g['query'][0]=g['reference'][0]
        with self.assertRaises(ValueError):
            r.predict_with_runtime(config(),self.context,Runtime(),g,self.root/'leak',Backend,source_kind='SYNTHETIC_FIXTURE')
    def test_timing_schedule_and_input_verification_boundary(self):
        inputs=group()['calibration'];events=[];ticks=iter(range(1000))
        def verify(x):events.append(('verify',x['global_identity']))
        def decode(x):events.append(('decode',x['global_identity']));return [np.zeros((1,1,3),np.uint8)]
        def sync():events.append(('sync',None))
        def clock():events.append(('clock',None));return next(ticks)
        values=r.measure_requests(Runtime(),inputs,decode,verify,sync,lambda:{},clock)
        self.assertEqual(len(values),30)
        self.assertEqual([x['phase'] for x in values].count('first'),1)
        self.assertEqual([x['phase'] for x in values].count('warmup'),2)
        self.assertEqual([x['phase'] for x in values].count('resident'),27)
        self.assertEqual([x[0] for x in events[:3]],['verify']*3)
        self.assertEqual([x['unit_id'] for x in values[3:]],['cal0','cal1','cal2']*9)
        self.assertTrue(all(x['total_request_s']==1 for x in values))
    def test_invalid_map_or_score_refused(self):
        with self.assertRaises(ValueError):
            r.normalize_output(dict(maps256=np.zeros((1,256,256),np.float64),object_score=1.0),'fixture',1)
        with self.assertRaises(ValueError):
            r.normalize_output(dict(maps256=np.zeros((1,256,256),np.float32),object_score=float('nan')),'fixture',1)
    def test_long_workspace_short_roots_and_prefix_collision(self):
        cfg=dict(config(),state_id='a'*64,config_id='b'*64)
        state=r._state_root(cfg,self.context)
        prediction=r._prediction_root(cfg,self.context)
        self.assertEqual(state.name,'a'*16)
        self.assertEqual(prediction.name,'b'*16)
        self.assertLess(len(str(state/'.t_12345678')),260)
        self.assertLess(len(str(prediction/'.t_12345678')),260)
        r.predict_with_runtime(cfg,self.context,Runtime(),group(),prediction,Backend,source_kind='SYNTHETIC_FIXTURE')
        collision=dict(cfg,config_id='b'*16+'c'*48)
        self.assertEqual(r._prediction_root(collision,self.context),prediction)
        with self.assertRaises(ValueError):
            r.predict_with_runtime(collision,self.context,Runtime(),group(),prediction,Backend,source_kind='SYNTHETIC_FIXTURE')


if __name__=='__main__':
    (RUN/'artifacts').mkdir(exist_ok=True)
    unittest.main(verbosity=2)
