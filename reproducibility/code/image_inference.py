"""Run image inference with explicitly supplied datasets and model weights.

Use plan or check-resources to inspect the fixed configurations without loading
a model. Inference rebuilds banks from reference images and the supplied normal
metric states. Execution requires current device-check evidence; see
docs/INFERENCE.md for the checks and resource bindings.
"""
from pathlib import Path
import argparse,hashlib,json,sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import dataset_bindings as p
import inference_runner as runner
ROOT=HERE.parent
def load_bindings(path):
    d=p.read(path);base=path.resolve().parent
    p.require(isinstance(d,dict) and d.get('schema')=='diream-explicit-reader-resources-v1','Bindings need schema diream-explicit-reader-resources-v1; copy RESOURCE_BINDINGS.example.json')
    p.require(isinstance(d.get('weights'),dict) and isinstance(d['weights'].get('anomalydino'),str) and bool(d['weights']['anomalydino'].strip()),'Bindings need a nonempty weights.anomalydino path')
    p.require(isinstance(d.get('image_sources',{}),dict) and all(isinstance(v,str) and v.strip() for v in d.get('image_sources',{}).values()),'image_sources must map dataset names to nonempty path strings')
    def resolved(x):return str(p.resolve_external(x,base))
    resources=dict(schema='explicit-resource-paths-v1',weights={'anomalydino':resolved(d['weights']['anomalydino'])},
        upstream={name:str(ROOT/'third_party'/folder) for name,folder in [('dinov2','DINOv2'),('anomalydino','AnomalyDINO'),('patchcore','PatchCore')]})
    for key in ['mhpc_torch_hub','subspace_giant']:
        if d.get(key):resources[key]=resolved(d[key])
    sources={k:resolved(v) for k,v in d.get('image_sources',{}).items()}
    return resources,sources
def check(resources,sources,category):
    errors=[];weight=Path(resources['weights']['anomalydino'])
    expected='b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'
    if not weight.is_file():errors.append(dict(resource='no-register ViT-S/14 weights',path=str(weight),expected_sha256=expected,obtain='https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth'))
    elif p.sha(weight)!=expected:errors.append(dict(resource='weights',reason='SHA256 differs',expected_sha256=expected))
    source='MANTA' if category=='paddy' else 'VisA'
    if source not in sources or not Path(sources[source]).exists():errors.append(dict(resource=source,reason='Bind original RGB archive or explicit extracted member directory',obtain='https://grainnet.github.io/MANTA' if source=='MANTA' else 'https://github.com/amazon-science/spot-diff'))
    for name,folder in resources['upstream'].items():
        if not Path(folder).is_dir():errors.append(dict(resource=name,path=folder,reason='Bundled source missing'))
    return errors
def main(argv=None):
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('command',choices=['plan','check-resources','fit','load','predict','resume','evaluate'])
    a.add_argument('--category',choices=['candle','capsules','cashew','chewinggum','fryum','macaroni1','macaroni2','pcb1','pcb2','pcb3','pcb4','pipe_fryum','paddy'])
    a.add_argument('--profile',choices=sorted(runner.PROFILES));a.add_argument('--budget',choices=['B','half_B','double_B','full'],default='B')
    a.add_argument('--bindings',type=Path);a.add_argument('--output',type=Path,help='Run directory outside the package; an existing matching run can be resumed')
    a.add_argument('--rebuild-evidence',type=Path,help='Current raw-rebuild bundle from public_rebuild_evidence.py')
    args=a.parse_args(argv);plan=runner.plan()
    if args.command=='plan':print(json.dumps(plan,indent=2));return 0
    p.require(args.category and args.profile,'Supply --category and --profile')
    configs=[c for c in plan['configs'] if c['category']==args.category and c['profile_id']==args.profile and c['budget_id']==args.budget]
    p.require(len(configs)==1,'Profile/category/budget combination is outside the fixed protocol')
    p.require(args.bindings is not None,'Supply --bindings RESOURCE_BINDINGS.json')
    resources,sources=load_bindings(args.bindings);errors=check(resources,sources,args.category)
    if errors:print(json.dumps(dict(status='MISSING_OR_CHANGED_EXPLICIT_RESOURCES',errors=errors),indent=2));return 2
    if args.command=='check-resources':print(json.dumps(dict(status='EXPLICIT_RESOURCES_PRESENT',scope='Weight SHA and resource existence; RGB members are checked during execution'),indent=2));return 0
    p.require(args.output is not None and args.rebuild_evidence is not None,'Raw operations require --output and --rebuild-evidence')
    output=args.output.resolve();p.require(not output.is_relative_to(ROOT),'Raw output must be outside package sources')
    context=p.defaults(dict(run_root=str(output),image_sources=sources,gpu_safety_checked=False))
    context.update(rebuild_evidence=str(args.rebuild_evidence.resolve()),rebuild_config_id=configs[0]['config_id'],bindings_path=str(args.bindings.resolve()))
    runner.validate_runtime=p.validate_runtime
    runner.validate_runtime(context,formal=True)
    output.mkdir(parents=True,exist_ok=True);resource_file=output/'EXPLICIT_RESOURCES.json'
    if resource_file.exists():p.require(p.read(resource_file)==resources,'Resources changed across a resumed run')
    else:resource_file.write_text(json.dumps(resources,indent=2)+'\n','utf8')
    context['resources_json']=str(resource_file);config=dict(configs[0],run_root=str(output),parent_run_root=str(p.PARENT))
    original=p.read(ROOT/'protocol/evaluation/original/EVALUATOR_SOURCES.json')
    gt_sources={}
    for category,source in original['gt_sources'].items():
        name='MANTA' if category=='paddy' else 'VisA'
        if name in sources:
            gt_sources[category]=dict(source,path=sources[name],kind='directory' if Path(sources[name]).is_dir() else source['kind'])
    evaluator_file=output/'protocol/EVALUATOR_SOURCES.json';evaluator_file.parent.mkdir(exist_ok=True)
    evaluator_file.write_text(json.dumps(dict(evaluator_sources=p.sources(),gt_sources=gt_sources),indent=2)+'\n','utf8')
    with runner.job_lock(output/'artifacts/GPU_SERIAL'):
        if args.command=='fit':result=runner.fit(config,context)
        elif args.command in ['predict','resume']:result=runner.predict(config,context)
        elif args.command=='evaluate':result=runner.evaluate(config,context)
        else:result=runner._load_for_science(config,context)[0]
    print(json.dumps(result,indent=2,default=str));return 0
if __name__=='__main__':raise SystemExit(main())
