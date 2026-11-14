"""Synthetic operator checks: NumPy fake tensors, no torch/model/GT execution."""
import ast
from contextlib import nullcontext
from pathlib import Path
import sys
import unittest

import numpy as np

run=Path(__file__).resolve().parents[2]
code=run/'code/directional_adapter'
sys.path.insert(0,str(code))
import directional_profiles as directional_adapter


def require(value,message):
    if not value:raise ValueError(message)


def extract_source_function(path,name,namespace,class_name=None):
    tree=ast.parse(path.read_text('utf8'))
    body=tree.body if class_name is None else next(item for item in tree.body if isinstance(item,ast.ClassDef) and item.name==class_name).body
    node=next(item for item in body if isinstance(item,ast.FunctionDef) and item.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]


normalize=extract_source_function(code/'reader_dino_encoder_source.py','normalize_components',dict(np=np,DIM=384,require=require))


class Tensor:
    """Minimal NumPy test double; it is not a scientific tensor backend."""
    def __init__(self,array):self.array=np.asarray(array)
    @property
    def shape(self):return self.array.shape
    @property
    def dtype(self):return self.array.dtype
    def __getitem__(self,item):return Tensor(self.array[item])
    def __truediv__(self,item):return Tensor(self.array/(item.array if isinstance(item,Tensor) else item))
    def __sub__(self,item):return Tensor(self.array-(item.array if isinstance(item,Tensor) else item))
    def __mul__(self,item):return Tensor(self.array*item)
    def to(self,device):
        if device!='cpu':raise ValueError('Synthetic test must remain CPU')
        return self
    def float(self):return Tensor(self.array.astype(np.float32))
    def cpu(self):return self
    def numpy(self):return self.array


class FakeTorch:
    float32=np.dtype(np.float32)
    @staticmethod
    def tensor(value,device,dtype):
        if device!='cpu':raise ValueError('Synthetic test must remain CPU')
        return Tensor(np.asarray(value,dtype=dtype))
    @staticmethod
    def from_numpy(value):return Tensor(value)
    @staticmethod
    def inference_mode():return nullcontext()
    @staticmethod
    def autocast(**kwargs):
        if kwargs!=dict(device_type='cpu',enabled=False):raise ValueError('Unexpected synthetic autocast')
        return nullcontext()


class SyntheticEncoder:
    def __init__(self):self.model=self;self.calls=[];self.outputs=[]
    def get_intermediate_layers(self,x,**kwargs):
        if kwargs!=dict(n=1,reshape=False,return_class_token=False,norm=True):raise ValueError('Encoder call contract changed')
        self.calls.append(x.array.copy())
        raw=np.zeros((1,1024,384),np.float32)
        raw[0,:,0]=(x.array[0,0]*100+100).ravel()
        raw[0,:,1]=2
        self.outputs.append(raw.copy())
        return [Tensor(raw)]


class EncodeTests(unittest.TestCase):
    def test_source_component_normalization_and_errors(self):
        raw=np.zeros((2,384),np.float32);raw[:,0]=3;raw[:,1]=4
        result,zeros=normalize(raw,'error')
        np.testing.assert_array_equal(result[:,:2],np.array([[.6,.8],[.6,.8]],np.float32))
        self.assertEqual(zeros,0)
        with self.assertRaises(ValueError):normalize(np.zeros((1,384),np.float32),'error')
        zero,count=normalize(np.zeros((1,384),np.float32),'keep')
        self.assertEqual(count,1);np.testing.assert_array_equal(zero,np.zeros((1,384),np.float32))
        with self.assertRaises(ValueError):normalize(np.full((1,384),np.nan,np.float32))

    def test_four_inverse_alignments_raw_masks_and_mean_one_search(self):
        image=np.zeros((32,32,3),np.uint8)
        image[...,0]=np.arange(1024,dtype=np.uint16).reshape(32,32)%251
        encoder=SyntheticEncoder();seen=[]
        def mask(raw,grid):
            seen.append(raw.copy())
            return raw[:,0]>100
        components,masks,counts=directional_adapter.encode_view(encoder,image,'M_W_FC4',square_views=lambda x:x,
            normalize_components=normalize,foreground_mask=mask,device='cpu',_torch_api=FakeTorch)
        self.assertEqual(counts['actual_encoder_calls'],4);self.assertEqual(counts['actual_pca_calls'],4)
        self.assertEqual(len(encoder.calls),4);self.assertEqual(len(seen),4)
        for k in range(4):
            # The mask receives exact raw tokens, before unit normalization.
            np.testing.assert_array_equal(seen[k],encoder.outputs[k][0])
            self.assertGreater(float(np.linalg.norm(seen[k][0])),1)
            expected,_=normalize(np.rot90(encoder.outputs[k].reshape(1,32,32,384),-k,axes=(1,2)).copy())
            np.testing.assert_array_equal(components[...,k,:],expected[0])
            np.testing.assert_array_equal(components[...,k,:],components[...,0,:])
            expected_mask=np.rot90((seen[k][:,0]>100).reshape(32,32),-k)
            np.testing.assert_array_equal(masks[...,k],expected_mask)
        calls=[]
        def search(q):calls.append(q.copy());return np.zeros(len(q),np.float32),None
        directional_adapter.raw_field(components,'M_W_FC4',search,dict(center=np.zeros(384),matrix=np.eye(384)),masks)
        self.assertEqual(len(calls),1);self.assertEqual(calls[0].shape,(1024,384))

    def test_single_reading_is_one_actual_synthetic_call(self):
        encoder=SyntheticEncoder();seen=[]
        x,masks,counts=directional_adapter.encode_view(encoder,np.full((32,32,3),128,np.uint8),'N_W_F0',square_views=lambda x:x,
            normalize_components=normalize,foreground_mask=lambda raw,grid:seen.append(raw.copy()) or np.ones(1024,bool),device='cpu',_torch_api=FakeTorch)
        self.assertEqual((len(encoder.calls),counts['actual_encoder_calls'],len(seen)),(1,1,1))
        self.assertEqual(x.shape,(32,32,384));self.assertEqual(masks.shape,(32,32,4))


class ForegroundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cv2
            from sklearn.decomposition import PCA
        except ImportError as exc:raise unittest.SkipTest(str(exc))
        import pinned_foreground_policy
        cls.primitive=staticmethod(pinned_foreground_policy.foreground_mask)
        cls.upstream=staticmethod(extract_source_function(code/'upstream_backbones_source.py','compute_background_mask',
                                 dict(np=np,cv2=cv2,PCA=PCA),class_name='DINOv2Wrapper'))

    def test_raw_token_pca_equals_pinned_upstream(self):
        for seed in [0,1,7]:
            features=np.random.default_rng(seed).normal(size=(64,32)).astype(np.float32)*12
            np.random.seed(0)
            expected=self.upstream(None,features,(8,8),threshold=10,masking_type=True)
            actual=self.primitive(features,(8,8))
            np.testing.assert_array_equal(actual,expected)
            self.assertEqual(actual.dtype,bool)

    def test_raw_token_scale_matters_and_rng_is_preserved(self):
        features=np.random.default_rng(2).normal(size=(64,32)).astype(np.float32)*20
        state=np.random.get_state();raw=self.primitive(features,(8,8));after=np.random.get_state()
        self.assertEqual(state[0],after[0]);np.testing.assert_array_equal(state[1],after[1]);self.assertEqual(state[2:],after[2:])
        unit=features/np.linalg.norm(features,axis=1,keepdims=True)
        normalized=self.primitive(unit,(8,8))
        self.assertTrue(raw.any());self.assertFalse(normalized.any())


if __name__=='__main__':unittest.main(verbosity=2)
