"""Explicit path binding for the unchanged, prospectively frozen raw driver.

Only filesystem bindings and frozen-file validation are adapted. Numerical
functions and scientific parameters remain those of unseen_validation.py.
Requires external data/weights/source/runtime; this is not scalar replay.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,shutil,sys
sys.dont_write_bytecode=True
HISTORICAL_ONLY={
 'provenance/category_inventory/all_composite_candidates.json',
 'provenance/category_inventory/agriculture_tar_inventory.json',
 'provenance/category_inventory/mechanics_tar_inventory.json',
 'study/data_sources/LOCAL_SEEN_AUDIT.json',
 'study/data_sources/CONSERVATIVE_CONSUMED_IDENTITY_UNION.json',
 'study/data_sources/CONSUMED_IDENTITY_APPENDIX.json'}

def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
def load(path): return json.loads(Path(path).read_text('utf-8-sig'))
def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bindings',type=Path,required=True)
    parser.add_argument('--records',type=Path,required=True,help='Saved FROZEN.json/MEMBERSHIP.json directory')
    parser.add_argument('--workdir',type=Path,required=True,help='Separate directory for actual raw outputs')
    parser.add_argument('--stage',choices=('check-bindings','check-imports','stage-rgb','encode','fit','predict','evaluate'),required=True)
    parser.add_argument('--category',choices=('soybean','terminal'))
    args=parser.parse_args();bindingpath=args.bindings.resolve();bindings=load(bindingpath)
    def resolve(value):
        path=Path(value);return (path if path.is_absolute() else bindingpath.parent/path).resolve()
    resource_root=resolve(bindings['resource_tree_root'])
    study=resolve(bindings.get('study_root',str(resource_root/'study')))
    reader=resolve(bindings.get('reader_root',str(resource_root/'reference_reader')))
    patchcore=resolve(bindings.get('patchcore_root',str(study/'baseline_sources/patchcore/patchcore-inspection-fcaa92f124fb1ad74a7acf56726decd4b27cbcad')))
    records=args.records.resolve();workdir=args.workdir.resolve()
    if workdir==records or (records in workdir.parents and args.stage not in ('check-bindings','check-imports')): raise ValueError('Use a separate work directory')
    driver=resolve(bindings.get('driver_source',str(Path(__file__).with_name('unseen_validation.py'))))
    f=load(records/'FROZEN.json');membership=records/'MEMBERSHIP.json'
    if sha(membership)!=f['membership_sha256']: raise ValueError('Frozen membership changed')
    scriptpin=next(row for row in f['source_and_code_pins'] if Path(row['path'].replace('\\','/')).name == 'unseen_validation.py')
    if sha(driver)!=scriptpin['sha256']: raise ValueError('Frozen scientific driver changed')
    recovery=resolve(bindings.get('resume_wrapper_source',str(Path(__file__).with_name('run_checked_stage.py'))))
    implementation=load(records/'IMPLEMENTATION_BINDING.json')
    if sha(recovery)!=implementation['resume_wrapper_sha256'] or sha(driver)!=implementation['driver_sha256']:
        raise ValueError('Implementation source differs from its recorded binding')
    overrides={key.replace('\\','/'):resolve(value) for key,value in bindings.get('pin_overrides',{}).items()}
    overrides[scriptpin['path'].replace('\\','/')]=driver
    def bound_pin_path(relative):
        if '/baseline_sources/patchcore/' in relative and relative.endswith('/src/patchcore/sampler.py'):
            target=patchcore/'src/patchcore/sampler.py'
        elif relative.startswith('study/'):
            target=study/relative.removeprefix('study/')
        elif relative.startswith('reference_reader/'):
            target=reader/relative.removeprefix('reference_reader/')
        else:
            return overrides.get(relative,resource_root/relative).resolve()
        target=target.resolve()
        if relative in overrides and overrides[relative]!=target:
            raise ValueError('Pin override conflicts with the bound resource root: '+relative)
        return target
    projected=[];resolved=[]
    history_mode=bindings.get('historical_evidence_mode','full_original_pins')
    if history_mode not in ('full_original_pins','readable_projection'):raise ValueError('Unknown historical evidence mode')
    if history_mode=='readable_projection':
        projection=load(records/'HISTORICAL_PIN_PROJECTION.json')
        history={row['frozen_relative_path']:row['original_sha256'] for row in projection['pins']}
        expected={row['path'].replace('\\','/'):row['sha256'] for row in f['source_and_code_pins'] if row['path'].replace('\\','/') in HISTORICAL_ONLY}
        if history!=expected or set(history)!=HISTORICAL_ONLY:raise ValueError('Historical projection differs from the exact six frozen historical-only pins')
    for pin in f['source_and_code_pins']:
        rel=pin['path'].replace('\\','/')
        if history_mode=='readable_projection' and rel in HISTORICAL_ONLY:
            projected.append({'frozen_relative_path':rel,'original_sha256':pin['sha256'],'role':'Historical category-selection provenance; fixed-membership calculation uses the saved selected members.'});continue
        path=bound_pin_path(rel)
        if not path.is_file() or sha(path)!=pin['sha256']:
            raise ValueError('Missing/changed explicit external dependency: '+pin['path'])
        resolved.append({'frozen_relative_path':pin['path'],'bound_file':str(path),'sha256':pin['sha256']})
    # A minimal external source tree may contain just these pinned paths plus
    # their import closure, including the official DINO source pinned by its
    # original UPSTREAM_FILE_MANIFEST. A complete old project is not required.
    archives=resolve(bindings['archives_root'])
    for pin in f['source_and_code_pins']:
        # Bindings may move entire directories, but each selected numerical
        # source must match its current public pin; original hashes are separate provenance.
        rel=pin['path'].replace('\\','/')
        if history_mode=='readable_projection' and rel in HISTORICAL_ONLY:continue
        mapped=bound_pin_path(rel)
        if not mapped.is_file() or sha(mapped)!=pin['sha256']:
            raise ValueError('Mapped numerical module differs from frozen source: '+str(mapped))
    workdir.mkdir(parents=True,exist_ok=True)
    record={'status':'PATH_BINDINGS_AND_PINNED_MODULES_CHECKED','stage':args.stage,
            'adaptation':'Path globals and frozen-file resolver only; no numerical function, profile, parameter, identity or mask rule changed.',
            'driver_sha256':sha(driver),'frozen_protocol_sha256':sha(records/'FROZEN.json'),
            'resume_wrapper_sha256':sha(recovery),'implementation_binding_sha256':sha(records/'IMPLEMENTATION_BINDING.json'),
            'bindings_sha256':sha(bindingpath),'resolved_pins':resolved,
            'historical_evidence_mode':history_mode,'historical_only_projected_pins':projected,
            'archive_payloads_not_checked_by_check_bindings':True,
            'note':'stage-rgb separately verifies both full archives. Model loader verifies DINO source/weight identities before actual CUDA inference. This check alone is not raw reproduction.',
            'raw_pipeline_executed':args.stage not in ('check-bindings','check-imports'),'GPU_calls_in_check_bindings':0}
    save(workdir/('BINDINGS_CHECK.json' if args.stage=='check-bindings' else 'BINDINGS_'+args.stage+'_'+str(args.category)+'.json'),record)
    if args.stage in ('check-bindings','check-imports'):
        if args.stage=='check-imports':
            for folder in (study/'operators',reader/'code',reader/'base/code',patchcore/'src'):sys.path.insert(0,str(folder))
            expected_modules={'dino_c4_response':study/'operators/dino_c4_response.py','host_original_search':study/'operators/host_original_search.py',
                              'normal_statistics':reader/'code/normal_statistics.py','coreset_runtime':reader/'code/coreset_runtime.py',
                              'reader_statistics_core':reader/'base/code/reader_statistics_core.py','reader_dino_matching':reader/'base/code/reader_dino_matching.py'}
            mappings=[]
            for name,path in expected_modules.items():
                imported=__import__(name)
                if Path(imported.__file__).resolve()!=path.resolve():raise ValueError('Imported numerical module from wrong resource directory')
                logical=next(row['frozen_relative_path'] for row in resolved if Path(row['bound_file'])==path.resolve())
                mappings.append({'module':name,'relative_file':logical,'sha256':sha(path)})
            identity=sys.modules['dino_c4_response'].verify_local_source(study/'baseline_sources')
            save(workdir/'IMPORT_MAPPING_CHECK.json',{'status':'PASS','modules':mappings,'DINO_source_entries_checked':identity['source_manifest_entries_checked'],
                 'DINO_commit':identity['code_commit'],'weight_sha256':identity['weight_sha256'],'model_constructed':False,'GPU_calls':0,'RGB_or_GT_access':False,
                 'note':'Computational import/resource closure checked in the supplied tree, not a full raw GPU rerun.'})
        print(json.dumps({'status':record['status'],'computational_pinned_files_checked':len(resolved),'historical_only_pins_projected':len(projected),'raw_pipeline_executed':False}));return
    for name in ('FROZEN.json','MEMBERSHIP.json','HISTORICAL_USAGE_AUDIT.json','IMPLEMENTATION_BINDING.json'):
        source=records/name;target=workdir/name
        if target.exists() and sha(target)!=sha(source): raise ValueError('Different existing run records')
        if not target.exists(): shutil.copyfile(source,target)
    spec=importlib.util.spec_from_file_location('unchanged_frozen_unseen_validation',driver)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.OUT=workdir;module.STUDY=study;module.READER=reader;module.CODE=reader/'code';module.BASECODE=reader/'base/code'
    module.PC=patchcore;module.ARCHIVES=archives
    def bound_frozen():
        current=load(workdir/'FROZEN.json')
        if sha(workdir/'FROZEN.json')!=sha(records/'FROZEN.json') or sha(workdir/'MEMBERSHIP.json')!=current['membership_sha256']:
            raise ValueError('Copied frozen protocol/membership differs')
        for row in resolved:
            if sha(Path(row['bound_file']))!=row['sha256']:raise ValueError('Bound source changed during raw run')
        return current
    module.frozen=bound_frozen
    repair_spec=importlib.util.spec_from_file_location('transparent_execution_recovery',recovery)
    repair=importlib.util.module_from_spec(repair_spec);repair_spec.loader.exec_module(repair)
    if args.stage in ('encode','fit','predict'):
        if not args.category:raise ValueError('Category required for this raw stage')
        repair.validate_encode_resume(module,args.category)
        if args.stage in ('fit','predict'):repair.validate_fit_resume(module,args.category)
        if args.stage=='predict':repair.validate_prediction_resume(module,args.category)
    if args.stage=='evaluate' and (workdir/'COMPLETE.json').exists():
        completed=load(workdir/'COMPLETE.json')
        if completed['freeze_sha256']!=sha(workdir/'FROZEN.json'):raise ValueError('Completed freeze changed')
        for row in completed['results']:
            if sha(workdir/row['path'])!=row['sha256']:raise ValueError('Completed evaluation payload changed')
        for category in module.CATEGORIES:repair.validate_prediction_resume(module,category,require_complete=True)
        print('EXISTING_BOUND_COMPLETED_EVALUATION_VERIFIED');return
    repair.install(module)
    sys.argv=[str(driver),args.stage]+(['--category',args.category] if args.category else [])
    module.main()
if __name__=='__main__':main()
