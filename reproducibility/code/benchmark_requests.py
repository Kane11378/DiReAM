"""Benchmark actual raw normal requests using explicit fitted reader models.

Fresh-process initialization, first request and resident repeats are separate.
This measures this portable driver and device; supplied observation records
describe their own implementation. No annotation or expected table is read.
Both explicit no-normal-rotation profiles time direct raw query requests with
their own newly fitted full bank; cached predictions are never timing inputs.
"""
from pathlib import Path
import sys,argparse,time,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('XFORMERS_DISABLED','1')
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from complete_methods import Runtime,METHODS,ALLOWED_PROFILES,read,dump,sha,image
from native_multiview import PROFILES as NATIVE_PROFILES,load_engines

def main():
    import torch,psutil,cv2
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=METHODS+NATIVE_PROFILES,required=True)
    p.add_argument('--profile',choices=ALLOWED_PROFILES)
    for name in ['base-reader','resources','manifest','image-source','model','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--category',required=True);p.add_argument('--units',type=int,default=3);a=p.parse_args()
    assert 1<=a.units<=3 and not a.output.exists(),'Use one to three normal units and a new output directory'
    if a.method=='DiReAM':assert a.profile is not None,'Select the single scientific profile being timed'
    else:assert a.profile is None
    spec=read(a.manifest);group=spec['groups'][a.category];rows=group['calibration'][:a.units]
    assert len(rows)==a.units and all(r['label_stratum']=='normal' for r in rows)
    fit=read(a.model/'FIT.json')
    assert fit['normal_ids']==[r['global_identity'] for r in group['reference']],'Fitted normal members do not match the requested benchmark configuration'
    assert fit.get('profile',fit.get('method'))==a.method,'Fitted method/profile does not match the request'
    if a.image_source.is_file():
        assert a.image_source.stat().st_size==spec['archive_bytes'] and sha(a.image_source)==spec['archive_sha256']
    else:assert a.image_source.is_dir()
    resources=read(a.resources);resources['_source_json']=str(a.resources.resolve())
    a.output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();torch.cuda.init();torch.cuda.synchronize();context=time.perf_counter()-start;start=time.perf_counter()
    native=a.method in NATIVE_PROFILES
    runtime=Runtime('AnomalyDINO_F' if native else a.method,a.base_reader,resources,[a.profile] if a.profile else None)
    if native:
        engines=load_engines(a.model,a.method);model=runtime.model
        def predict(vv):
            maps=[];scores=[]
            for v,im in enumerate(vv):
                f,grid=model._features(im);q=model._normalize(f);e=engines[v] if len(engines)>1 else engines[0]
                d=e.run(1,q)[0][:,0]/2;k=int(len(d)*.01);scores.append(float(d.max() if k==0 else np.mean(np.sort(d)[::-1][:k])))
                field=model.upstream_utils.dists2map(d.reshape(grid),im.shape).astype(np.float32)
                maps.append(cv2.resize(field,(256,256),interpolation=cv2.INTER_LINEAR))
            return dict(maps256=np.stack(maps),object_score=np.array(max(scores),np.float64),actual_readings_per_view=np.array(1,np.int32))
    else:runtime.load(a.model);predict=runtime.predict
    torch.cuda.synchronize();initialization=time.perf_counter()-start
    init_memory=dict(cuda_allocated=torch.cuda.max_memory_allocated(),cuda_reserved=torch.cuda.max_memory_reserved(),process_rss=psutil.Process().memory_info().rss)
    observations=[]
    def request(i,phase):
        torch.cuda.synchronize();t=time.perf_counter();vv=image(rows[i],a.image_source);fields=predict(vv);torch.cuda.synchronize();elapsed=time.perf_counter()-t
        memory=psutil.Process().memory_info()
        row=dict(phase=phase,unit_index=i,global_identity=rows[i]['global_identity'],actual_views=len(vv),total_request_s=elapsed,rss_bytes=memory.rss,process_lifetime_peak_rss_bytes=getattr(memory,'peak_wset',memory.rss))
        assert all(np.isfinite(v).all() for v in fields.values()),'Nonfinite returned output'
        path=a.output/'raw_outputs'/f'{len(observations):03d}.npz';path.parent.mkdir(exist_ok=True)
        np.savez(path,**fields);row.update(output=path.name,output_sha256=sha(path));observations.append(row)
    request(0,'first')
    for i in [0,min(1,a.units-1)]:request(i,'warmup')
    torch.cuda.reset_peak_memory_stats()
    for cycle in range(9):
        for i in range(a.units):request(i,'resident_warm')
    warm=np.array([r['total_request_s'] for r in observations if r['phase']=='resident_warm'])
    # Output persistence, finite-value checks and hashes are outside each timer.
    # Source-image bytes are verified and decoded within each raw request.
    result=dict(status='COMPLETE_PORTABLE_RAW_REQUEST_BENCHMARK',method=a.method,profile=a.profile,category=a.category,manifest_sha256=sha(a.manifest),model_fit_sha256=sha(a.model/'FIT.json'),cuda_context_s=context,initialization_s=initialization,initialization_memory=init_memory,first_request_s=observations[0]['total_request_s'],resident_warm_n=len(warm),resident_warm_mean_s=float(warm.mean()),resident_warm_median_s=float(np.median(warm)),resident_warm_p95_s=float(np.quantile(warm,.95)),resident_peak_cuda_allocated=torch.cuda.max_memory_allocated(),resident_peak_cuda_reserved=torch.cuda.max_memory_reserved(),samples=observations,normal_cost_units=a.units,GT_reads=0,scope='Actual explicit RGB bytes to field/native score with resident model. First request, model/resource initialization and warm requests are distinct. Outputs persist outside request timing. OS file cache is uncontrolled; repeats are workload observations, not efficacy samples. This portable-driver timing does not replace the separately supplied implementation-specific cost records.')
    if a.method in ['AnomalyDINO_F_norot','AnomalyDINO_U_pool_norot']:
        result['normal_configuration']={k:fit[k] for k in ['normal_rotation','normal_augmentations_per_view','normal_input_path','memory','query_foreground','reference_foreground','normal_vectors','normal_shards_sha256']}
    dump(a.output/'RESULT.json',result);print(result['status'],flush=True)
if __name__=='__main__':main()
