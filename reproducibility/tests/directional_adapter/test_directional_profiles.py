"""Synthetic numerical/operator tests only; no scientific samples or AP."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

run = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(run / 'code/directional_adapter'))
import directional_profiles as directional_adapter


def model():
    return dict(center=np.zeros(384, np.float64), matrix=np.eye(384, dtype=np.float64))


class Capture:
    def __init__(self, values=None):
        self.queries = []
        self.values = values

    def __call__(self, x):
        self.queries.append(x.copy())
        value = np.zeros(len(x), np.float32) if self.values is None else np.resize(np.asarray(self.values, np.float32), len(x))
        return value, np.zeros(len(x), np.int64)


class Profiles(unittest.TestCase):
    def test_unknown_profile_rejected(self):
        for value in ('within_I_typo', 'within_I01', '', None, 'N_W_FC4'):
            with self.assertRaises(ValueError):
                directional_adapter.profile(value)

    def test_profile_reading_and_query_counts(self):
        for name, readings, queries in [('N_W_U',1,1),('I_W_U',4,4),('M_W_U',4,1),
                                        ('N_W_F0',1,1),('I_W_FC4',4,4),('M_W_FC4',4,1)]:
            item = directional_adapter.profile(name)
            self.assertEqual((item.encoder_readings_per_view,item.distance_queries_per_anchor),(readings,queries))

    def test_mean_fp64_then_fp32_without_normalization(self):
        x = np.zeros((32,32,4,384), np.float32)
        x[:,:,:,0] = [1e8,1,-1e8,1]
        engine = Capture()
        directional_adapter.raw_field(x,'M_W_U',engine,model())
        self.assertEqual(engine.queries[0].dtype,np.float32)
        np.testing.assert_array_equal(engine.queries[0][:,0], np.full(1024,.5,np.float32))
        # Re-normalizing the mean would change .5 to 1.
        self.assertEqual(float(np.linalg.norm(engine.queries[0][0])), .5)

    def test_transform_fp64_then_fp32_without_normalization(self):
        x = np.zeros((32,32,384), np.float32); x[...,0] = 2
        fitted=model(); fitted['matrix'][0,0]=3; fitted['center'][0]=.25
        engine=Capture()
        directional_adapter.raw_field(x,'N_W_U',engine,fitted)
        np.testing.assert_array_equal(engine.queries[0][:,0],np.full(1024,5.25,np.float32))

    def test_n_i_share_same_unrestricted_engine_and_donor(self):
        x=np.zeros((32,32,4,384),np.float32); x[...,0]=[3,4,5,6]
        # Candidate row 2 may originate from ANY normal digital direction.
        bank=np.zeros((4,384),np.float32); bank[:,0]=[-10,-20,3,8]
        class Search:
            def __init__(self):self.calls=[]
            def __call__(self,q):
                self.calls.append(q.copy())
                d=((q[:,None,:].astype(np.float64)-bank)**2).sum(-1)/384
                return d.min(-1).astype(np.float32),d.argmin(-1)
        engine=Search()
        n=directional_adapter.raw_field(x,'N_W_U',engine,model())[0]
        i=directional_adapter.raw_field(x,'I_W_U',engine,model())[0]
        np.testing.assert_array_equal(n,np.zeros((32,32),np.float32))
        np.testing.assert_allclose(i,np.full((32,32),(0+1+4+4)/8,np.float32))
        self.assertEqual([len(v) for v in engine.calls],[1024,4096])

    def test_i_min_before_equal_mean(self):
        x=np.zeros((32,32,4,384),np.float32)
        engine=Capture([1,2,3,10])
        result=directional_adapter.raw_field(x,'I_W_U',engine,model())[0]
        np.testing.assert_array_equal(result,np.full((32,32),4*192,np.float32))
        self.assertEqual(engine.queries[0].shape,(4096,384))

    def test_fc4_i_and_m_keep_distinct_mask_rules(self):
        x=np.zeros((32,32,4,384),np.float32)
        masks=np.broadcast_to(np.array([True,False,True,False]),(32,32,4)).copy()
        independent=directional_adapter.raw_field(x,'I_W_FC4',Capture([1,2,3,4]),model(),masks)[0]
        mean=directional_adapter.raw_field(x,'M_W_FC4',Capture([2.5]),model(),masks)[0]
        np.testing.assert_array_equal(independent,np.full((32,32),192,np.float32))
        np.testing.assert_array_equal(mean,np.full((32,32),240,np.float32))

    def test_all_zero_masks_remain_zero(self):
        x=np.ones((32,32,4,384),np.float32); masks=np.zeros((32,32,4),bool)
        for profile in ['N_W_F0','I_W_FC4','M_W_FC4']:
            result=directional_adapter.raw_field(x,profile,Capture([2]),model(),masks)[0]
            np.testing.assert_array_equal(result,np.zeros((32,32),np.float32))

    def test_half_squared_distance_scaling(self):
        x=np.zeros((32,32,384),np.float32); x[...,0]=2
        def search(q):return (q*q).sum(-1)/np.float32(384),None
        result=directional_adapter.raw_field(x,'N_W_U',search,model())[0]
        np.testing.assert_allclose(result,np.full((32,32),2,np.float32),rtol=1e-7)

    def test_shared_state_must_be_literal_identity(self):
        binding=directional_adapter.StateBinding(0,'component','b','bh','i','ih','m','mh','members',10)
        directional_adapter.validate_shared_component(binding,binding)
        for changed in [replace(binding,bank_sha256='different'),replace(binding,indices_sha256='different'),
                        replace(binding,model_sha256='different'),replace(binding,retained_rows=20),
                        replace(binding,slot_id=1),replace(binding,bank_kind='mean')]:
            with self.assertRaises(ValueError):directional_adapter.validate_shared_component(binding,changed)

    def test_bad_inputs_fail_before_search(self):
        x=np.zeros((32,32,4,384),np.float32)
        for value in [x.astype(np.float64),np.zeros((32,32,3,384),np.float32),np.full_like(x,np.nan)]:
            with self.assertRaises(ValueError):directional_adapter.raw_field(value,'I_W_U',Capture(),model())
        with self.assertRaises(ValueError):directional_adapter.raw_field(x,'M_W_FC4',Capture(),model(),np.zeros((32,32,4),np.float32))


if __name__=='__main__':
    unittest.main(verbosity=2)
