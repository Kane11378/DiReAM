"""AnomalyDINO agnostic VisA query foreground policy.

The PCA/sign/morphology operations reproduce DINOv2Wrapper.compute_background_mask
in AnomalyDINO commit b9d1c2648e3a5247437d4d953d907a8f3d994457.
References remain unmasked, as in the upstream CLI default. This module does
not choose foreground policies from labels or modify the normal memory.
"""
import numpy as np

def foreground_mask(features,grid):
    import cv2
    from sklearn.decomposition import PCA
    if features.ndim!=2 or len(features)!=int(np.prod(grid)):
        raise ValueError('Raw token features and their spatial grid are required')
    state=np.random.get_state()
    try:
        np.random.seed(0)
        first_pc=PCA(n_components=1,svd_solver='randomized').fit_transform(features.astype(np.float32))
    finally:np.random.set_state(state)
    mask=first_pc>10
    center=mask.reshape(grid)[int(grid[0]*.2):int(grid[0]*.8),int(grid[1]*.2):int(grid[1]*.8)]
    if center.sum()<=center.size*.35:mask=-first_pc>10
    # Preserve the upstream operation on the flattened Nx1 mask, including shape.
    mask=cv2.dilate(mask.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    mask=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8)).astype(bool)
    return mask.squeeze()

def masked_turn(model,image,raw_distances):
    features,grid=model._features(image)
    if raw_distances.shape!=tuple(grid):raise ValueError('Cached distance grid differs from fresh feature grid')
    mask=foreground_mask(features,grid)
    distances=raw_distances.astype(np.float64).ravel();distances[~mask]=0
    count=int(len(distances)*.01)
    score=float(np.max(distances) if count==0 else np.mean(np.sort(distances)[::-1][:count]))
    field=model.upstream_utils.dists2map(distances.reshape(grid),image.shape).astype(np.float32)
    return field,score,mask

def combine(aligned,scores):
    import cv2
    if len(aligned)!=4 or len(scores)!=4:raise ValueError('All four fixed query turns are required')
    base=cv2.resize(aligned[0],(256,256),interpolation=cv2.INTER_LINEAR)[None].astype(np.float32)
    average=np.mean(np.stack(aligned).astype(np.float64),axis=0).astype(np.float32)
    c4=cv2.resize(average,(256,256),interpolation=cv2.INTER_LINEAR)[None].astype(np.float32)
    return dict(base=base,c4=c4,base_score=np.array(scores[0]),c4_score=np.array(np.mean(scores)),turn_scores=np.asarray(scores,np.float64))

def main():
    import argparse,json
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--self-check',action='store_true',required=True);p.parse_args()
    rng=np.random.default_rng(0);f=rng.normal(size=(64,32)).astype(np.float32)*12
    a=foreground_mask(f,(8,8));b=foreground_mask(f,(8,8))
    assert np.array_equal(a,b) and a.shape==(64,) and a.dtype==bool
    x=np.arange(64,dtype=np.float32).reshape(8,8)
    out=combine([x,x,x,x],[1.,1.,1.,1.])
    assert np.array_equal(out['base'],out['c4']) and float(out['c4_score'])==1
    print(json.dumps({'generated_input_check':'PASS','mask_retained':int(a.sum()),'mask_total':len(a),'scientific_performance_measured':False}))
if __name__=='__main__':main()
