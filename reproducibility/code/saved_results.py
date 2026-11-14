"""Recompute statistics from saved per-unit observations.

Inputs contain the original observations and their integrity records. Derived
tables are read after computation for full-precision comparison. Each command
writes to a new directory outside this package. Image inference is separate.
"""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,math,shutil,statistics,sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import numpy as np
import evaluation as ev
import dataset_bindings as portable
ROOT=HERE.parent
DATA=ROOT/'results/evaluation'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return portable.sha(p)
def require(ok,msg):
    if not ok:raise ValueError(msg)
def write(p,value):p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n','utf8')
def writecsv(p,rr):
    with p.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rr for k in r)));w.writeheader();w.writerows(rr)
def verify():
    from public_projection import verify as verify_projection
    projection=verify_projection(ROOT)
    pins=read(ROOT/'protocol/evaluation/SAVED_INPUTS.json')['files']
    for p,h in pins.items():require(sha(ROOT/p)==h,'Bound saved input changed: '+p)
    require(len(rows(DATA/'RESULTS_CATEGORY.csv'))==113,'Category count changed')
    require(len(rows(DATA/'RESULTS_UNITS.csv'))==19560,'Whole-unit row count changed')
    directories=sorted((DATA/'e').iterdir());require(len(directories)==113,'Evaluation count changed')
    for directory in directories:
        seal,summary,units=ev.validate_evaluation_complete(directory)
        require(seal['status']=='COMPLETE_EVALUATION','Synthetic/nonproduction evaluation cannot enter replay')
        require(summary['state_sha256']==seal['config']['state_sha256'],'Evaluation state mismatch')
        group=portable.role_group(seal['config'])
        require([u['global_identity'] for u in units]==[u['global_identity'] for u in group['query']],'Evaluation query member order differs')
    return dict(bound_files=len(pins),evaluations=113,category_rows=113,whole_unit_rows=19560,public_projection=projection)
def compare_numeric(actual,expected,key,numbers,tol=1e-12):
    a={tuple(str(r[k]) for k in key):r for r in actual};b={tuple(str(r[k]) for k in key):r for r in expected}
    require(set(a)==set(b),'Derived row identities differ');errors=[]
    for ident,row in a.items():
        for name in numbers:
            x,y=float(row[name]),float(b[ident][name]);err=abs(x-y);errors.append(err)
            require(math.isclose(x,y,rel_tol=tol,abs_tol=tol),f'Derived value differs: {ident} {name}: {x} vs {y}')
    return dict(rows=len(a),numeric_cells=len(errors),max_abs_error=max(errors,default=0),relative_and_absolute_tolerance=tol)
def macros(out):
    rr=rows(DATA/'RESULTS_CATEGORY.csv');result=[]
    complete_profiles=read(ROOT/'protocol/evaluation/REPLAY_DESIGN.json')['complete_profiles']
    for profile,budget in complete_profiles:
      for stratum in ['all12','prior_used6','remaining6']:
        cats=ev._categories_for_requirement(dict(category_or_macro=stratum))
        selected=[r for r in rr if r['cohort_id']=='visa12' and r['profile_id']==profile and r['budget_id']==budget and r['category'] in cats]
        require(sorted(r['category'] for r in selected)==sorted(cats),'Missing or duplicate fixed macro category')
        z=dict(cohort_id='visa12',stratum=stratum,profile_id=profile,budget_id=budget,categories_n=len(cats))
        for metric in ev.FIVE_ENDPOINTS:z[metric]=statistics.fmean(float(r[metric]) for r in selected)
        result.append(z)
    expected=rows(DATA/'COMPLETE_MACROS.csv')
    report=compare_numeric(result,expected,['cohort_id','stratum','profile_id','budget_id'],list(ev.FIVE_ENDPOINTS)+['categories_n'])
    writecsv(out/'COMPLETE_MACROS.csv',result);return report
def paired(out):
    req=rows(ROOT/'protocol/evaluation/requirements.csv')
    require(len(req)==69 and len({r['logical_contrast_id'] for r in req})==68,'69/68 contrast identity changed')
    # Check ordered units and endpoint eligibility before paired resampling.
    result=ev._bootstrap_impl(sorted((DATA/'e').iterdir()),req,portable.sources(),out/'paired',False)
    actual=rows(out/'paired/PAIRED_CONTRASTS.csv');expected=rows(DATA/'bootstrap/PAIRED_CONTRASTS.csv')
    key=['cohort_id','category_or_macro','left_profile','right_profile','left_budget','right_budget','endpoint']
    report=compare_numeric(actual,expected,key,['difference_pp','ci_low_pp','ci_high_pp','paired_units_n','categories_n','bootstrap_n','bootstrap_seed'])
    bykey={tuple(r[k] for k in key):r for r in actual}
    expanded=[dict(r,**{k:bykey[tuple(r[k] for k in key)][k] for k in ['difference_pp','ci_low_pp','ci_high_pp','paired_units_n','categories_n','bootstrap_n','bootstrap_seed']}) for r in req]
    compare_numeric(expanded,rows(DATA/'PAIRED_CONTRASTS.csv'),['requirement_id'],['difference_pp','ci_low_pp','ci_high_pp'])
    writecsv(out/'PAIRED_CONTRASTS_69.csv',expanded)
    report.update(logical_requirements=69,unique_estimates=68,bootstrap_replicates=10000,seed=20261022)
    return report
def costs(out):
    p=ROOT/'code/request_statistics.py'
    spec=importlib.util.spec_from_file_location('_saved_request_statistics',p);agg=importlib.util.module_from_spec(spec);spec.loader.exec_module(agg)
    configs={c['config_id']:c for c in read(ROOT/'protocol/evaluation/original/RUN_PLAN.json')['configs']}
    actual=[];request_n=0
    for d in sorted((DATA/'t').iterdir()):
        doc=read(d/'REQUESTS.json');c=configs[doc['config_id']];g=portable.role_group(c)
        item=agg.read_timing(d,c,doc['state_sha256'],g);require(item is not None,'Incomplete timing source')
        s=item['summary'];actual.append(s);request_n+=len(item['requests'])
    report=compare_numeric(actual,rows(DATA/'TIMING_SUMMARY.csv'),['category','profile_id','budget_id'],['first_request_s','resident_median_s','resident_p95_s','resident_n','cuda_allocated_peak_bytes','cuda_reserved_peak_bytes','feature_bytes','search_norm_bytes','search_origin_bytes','device_feature_bytes'])
    require(request_n==750 and len(actual)==25,'Request/config totals differ')
    for s in actual:
        q=[r for r in rows(DATA/'RESULTS_CATEGORY.csv') if r['category']==s['category'] and r['profile_id']==s['profile_id'] and r['budget_id']==s['budget_id']]
        require(len(q)==1 and q[0]['state_sha256']==s['state_sha256'],'Quality/timing state mismatch')
        target=next(r for r in rows(DATA/'TIMING_SUMMARY.csv') if all(r[k]==str(s[k]) for k in ['category','profile_id','budget_id']))
        require(abs(float(q[0]['unit_ap'])-float(target['category_unit_ap']))<1e-12,'Timing quality-link AP differs')
    writecsv(out/'TIMING_SUMMARY.csv',actual)
    report.update(requests=750,configurations=25,cuda_peak_scope='Cumulative initialization + first + warmup + resident full window; historical series remains separate')
    return report
def budgets(out):
    rr=rows(DATA/'STATE_BUDGETS.csv');require(len(rr)==96,'Budget row count changed')
    by={}
    for r in rr:
        require(int(r['feature_dim'])==384 and r['feature_dtype']=='float32','Feature geometry changed')
        require(int(r['feature_bytes'])==int(r['retained_rows'])*384*4,'FP32 bank payload differs')
        require(int(r['index_metadata_bytes'])==int(r['retained_rows'])*8,'Selected-index payload differs')
        require(int(r['retained_rows'])<=int(r['candidate_rows']),'Retained budget exceeds candidates')
        by.setdefault(r['state_id'],[]).append(r)
    for sid,group in by.items():
        state=read(ROOT/'protocol/evaluation/state_metadata'/sid[:16]/'STATE.json')
        require(state['state_id']==sid and all(r['state_sha256']==state['state_identity_sha256'] for r in group),'State metadata binding differs')
        expected=state['slots'] if state['profile_family']=='diream' else state['native_state']['states']
        require([int(r['retained_rows']) for r in group]==[int(r['retained_rows']) for r in expected],'Per-slot/pooled K differs from saved state')
        require(len(group)==len(expected),'Actual slot topology differs')
    for s in rows(DATA/'TIMING_SUMMARY.csv'):
        group=[r for r in rr if r['state_sha256']==s['state_sha256']]
        require(sum(int(r['feature_bytes']) for r in group)==int(s['device_feature_bytes']),'Loaded payload differs from saved state')
        require(sum(int(r['retained_rows'])*4 for r in group)==int(s['search_norm_bytes']),'Search norm payload differs')
    writecsv(out/'STATE_BUDGETS.csv',rr)
    return dict(rows=96,physical_states=len(by),scope='Saved metadata, slot counts and array payload geometry; large arrays not opened or rehashed')
def export(out):
    copied=[]
    for folder in ['results','protocol','proposed_figures/plot_data','proposed_figures/fixed_cases','figure_sources/shared/data','figures','extensions/timing/records','extensions/unseen/records']:
        for source in (ROOT/folder).rglob('*'):
            if source.is_file():
                relative=source.relative_to(ROOT);target=out/relative
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source,target)
                copied.append(dict(file=relative.as_posix(),sha256=sha(target)))
    for name in ['NOTICE.md','figure_data_index.json','docs/DATA_SOURCES.md','docs/STATISTICS.md','docs/MANTA_CREDITS.md','docs/MANTA_MATERIALS.csv']:
        source=ROOT/name;target=out/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        copied.append(dict(file=name,sha256=sha(target)))
    navigation=out/'README.md'
    navigation.write_text('# Saved observations and analysis data\n\nResults and per-unit observations are under results/. Protocols identify the fixed populations, methods and analysis settings. Figure tables are under proposed_figures/plot_data/ and fixed examples under proposed_figures/fixed_cases/. Complete numerical figure inputs are under figure_sources/shared/data/ and available final figures under figures/. Extension observations are included under extensions/. Run analysis commands from the original reproducibility package.\n','utf8')
    copied.append(dict(file='README.md',sha256=sha(navigation)))
    require((out/'results/evaluation/RESULTS_CATEGORY.csv').is_file(),'Fixed-panel results absent')
    return dict(files=copied,status='SCIENCE_DATA_EXPORTED')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['verify','macros','paired','costs','budgets','export','all']);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    output=a.output.resolve();require(not output.exists(),'Output directory already exists; choose a new directory')
    require(not output.is_relative_to(ROOT),'Output must be outside the immutable reader tree')
    verified=verify();output.mkdir(parents=True)
    report=dict(scope='DETERMINISTIC_SAVED_SCALAR_CPU_REPLAY',verification=verified,python=sys.version,numpy=np.__version__,imported_evaluator=ev.__file__,root=str(ROOT))
    selected=['macros','paired','costs','budgets','export'] if a.command=='all' else ([] if a.command=='verify' else [a.command])
    for name in selected:report[name]=globals()[name](output)
    write(output/'REPLAY_REPORT.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
