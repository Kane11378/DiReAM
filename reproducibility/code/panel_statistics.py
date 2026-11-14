"""Paired whole-unit summaries from actual field-evaluation CSVs.

No expected-table input, method selection, fitting, pixel inference or private
workspace lookup. Fixed-category conditional pointwise uncertainty only.
"""
from pathlib import Path
import json,csv,argparse,math
import numpy as np
def read(p):return json.loads(Path(p).read_text("utf-8-sig"))
def rows(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def write(p,rr):
 p.parent.mkdir(parents=True,exist_ok=True);keys=list(dict.fromkeys(k for r in rr for k in r))
 with p.open('w',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rr)

def number(x):return float(x) if x not in [None,'','None'] else None

def yes(x):return x in [True,1,'True','true','1']

def contrasts(units,groups,comparisons,cohort):
 index={}
 for r in units:index.setdefault((r['category'],r['method']),{})[r['global_identity']]=r
 out=[]
 for group,categories in groups.items():
  for left,right in comparisons:
   for endpoint in ['object_pixel_AP','coverage_0.01','coverage_0.05','gated_coverage_0.01','gated_coverage_0.05','native_alarm_positive','native_alarm_normal']:
    pairs=[];used=[]
    for cat in categories:
     a=index.get((cat,left));b=index.get((cat,right))
     if a is None or b is None:continue
     assert set(a)==set(b),(cat,left,right,'unit mismatch');diff=[]
     for identity in sorted(a):
      x,y=a[identity],b[identity];assert yes(x['object_anomalous'])==yes(y['object_anomalous'])
      if endpoint.startswith('native_alarm'):
       want=endpoint.endswith('positive')
       if yes(x['object_anomalous'])!=want:continue
       u,v=number(x['native_alarm']),number(y['native_alarm'])
      else:
       u,v=number(x.get(endpoint)),number(y.get(endpoint))
       assert (u is None)==(v is None),(identity,endpoint)
       if u is None:continue
      diff.append(u-v)
     assert diff,(cat,endpoint);pairs.append(np.asarray(diff,np.float64));used.append(cat)
    if len(used)!=len(categories):continue
    # Conditional uncertainty across units of the specified fixed categories.
    # Categories are never resampled as purported independent random replicas.
    rng=np.random.default_rng(20261022);draw=np.zeros(10000,np.float64)
    for d in pairs:draw+=d[rng.integers(0,len(d),(10000,len(d)))].mean(1)/len(pairs)
    point=float(np.mean([d.mean() for d in pairs]));lo,hi=np.quantile(draw,[.025,.975])
    out.append(dict(cohort=cohort,stratum=group,left=left,right=right,endpoint=endpoint,categories=';'.join(used),category_n=len(used),eligible_units=sum(map(len,pairs)),difference=point,pointwise95_lo=float(lo),pointwise95_hi=float(hi),bootstrap_replicates=10000,bootstrap_seed=20261022,inference='Paired whole-unit within fixed categories; equal-category macro; descriptive pointwise intervals, no multiplicity correction or category-population inference'))
 return out

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();spec=read(a.inputs);root=a.inputs.resolve().parent;summary=[];units=[]
 for entry in spec['evaluations']:
  summary+=rows((root/entry['summary']).resolve());units+=rows((root/entry['per_unit']).resolve())
 keys=[(r['category'],r['method']) for r in summary];assert len(keys)==len(set(keys)),'Duplicate summary rows'
 keys=[(r['category'],r['method'],r['global_identity']) for r in units];assert len(keys)==len(set(keys)),'Duplicate whole units'
 groups=spec['strata'];primary=spec.get('primary');comparisons=spec.get('comparisons',[(primary,m) for m in spec.get('comparators',[])]);paired=[];macro=[]
 for name,categories in groups.items():
  fixed=spec.get('comparisons_by_stratum',{}).get(name,comparisons)
  assert fixed and all(len(pair)==2 and all(isinstance(x,str) and x for x in pair) for pair in fixed),'Supply explicit procedure pairs'
  paired+=contrasts(units,{name:categories},fixed,spec.get('cohort','explicit field recomputation'))
 metrics=['mean_object_pixel_AP','pooled_pixel_AP','pooled_pixel_AUROC','object_AP','object_AUROC','sensitivity','false_alarm_rate','coverage_0.01','coverage_0.05','gated_coverage_0.01','gated_coverage_0.05']
 for name,categories in groups.items():
  for method in sorted({r['method'] for r in summary}):
   rr=[r for r in summary if r['method']==method and r['category'] in categories]
   if len(rr)!=len(categories):continue
   z=dict(stratum=name,method=method,categories=len(categories));z.update({k:float(np.mean([float(r[k]) for r in rr])) for k in metrics});macro.append(z)
 a.output.mkdir(parents=True,exist_ok=False);write(a.output/'MACRO.csv',macro);write(a.output/'PAIRED_CONTRASTS.csv',paired)
 (a.output/'COMPLETE.json').write_text(json.dumps(dict(status='ACTUAL_FIELD_SCALAR_STATISTICS_COMPLETE',bootstrap_replicates=10000,bootstrap_seed=20261022,expected_tables_used_as_inputs=False,scope='Paired whole units within each specified fixed category. Equal-category macro. Descriptive pointwise95percent intervals, no multiple-comparison adjustment/category-population inference.'),indent=2),encoding='utf8')
if __name__=='__main__':main()
