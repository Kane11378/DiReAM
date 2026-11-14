"""Recompute new-category Unit AP, native image metrics and paired contrasts.

This portable scalar replay reads PER_UNIT.csv and FROZEN.json. It does not
reconstruct pooled-pixel metrics, which require the saved spatial fields/GT.
No model, image, mask, GPU, network, or raw-inference runtime is accessed.
"""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np

def table(path,rows):
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)

def binary(labels,scores):
    labels=np.asarray(labels,bool);scores=np.asarray(scores,np.float64)
    order=np.argsort(-scores,kind='stable'); y=labels[order];s=scores[order]
    ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1]
    tp=np.cumsum(y,dtype=np.int64)[ends];fp=ends+1-tp
    dt=np.diff(np.r_[0,tp]);df=np.diff(np.r_[0,fp]);pos=int(labels.sum());neg=len(labels)-pos
    return {'Image_AP':float(np.sum(dt/(pos or 1)*tp/(ends+1))) if pos else None,
            'Image_AUROC':float(np.sum(dt*(neg-fp+.5*df))/(pos*neg)) if pos and neg else None}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();evidence=args.evidence.resolve();out=args.output.resolve()
    if out==evidence or evidence in out.parents: raise ValueError('Choose a separate replay output directory')
    if out.exists(): raise ValueError('Output directory already exists; preserve previous replay')
    freeze=json.loads((evidence/'FROZEN.json').read_text('utf-8-sig'))
    with (evidence/'PER_UNIT.csv').open(encoding='utf-8-sig',newline='') as stream: rows=list(csv.DictReader(stream))
    categories=freeze['categories'];profiles=freeze['profiles'];pairs=freeze['contrasts'];values={};summaries=[]
    for cat in categories:
        values[cat]={}
        for name in profiles:
            rr=[row for row in rows if row['category']==cat and row['profile']==name]
            eligible=[row for row in rr if row['Unit_AP_eligible']=='True']
            ap=np.array([float(row['Unit_AP']) for row in eligible],np.float64);values[cat][name]=ap
            bad=np.array([row['anomalous']=='True' for row in rr]);alarm=np.array([int(row['alarm']) for row in rr],bool)
            summary={'category':cat,'profile':name,'query_units':len(rr),'Unit_AP_eligible':len(ap),'Unit_AP':float(ap.mean()),
                     **binary(bad,[float(row['native_score']) for row in rr]),
                     'sensitivity':float(alarm[bad].mean()),'FAR':float(alarm[~bad].mean()),'false_alarms':int(alarm[~bad].sum())}
            for key in ('coverage_0.01','coverage_0.05','gated_coverage_0.01','gated_coverage_0.05'):
                summary[key]=float(np.mean([float(row[key]) for row in eligible]))
            summaries.append(summary)
            if name==profiles[0]: identifiers=[row['global_identity'] for row in eligible]
            elif identifiers!=[row['global_identity'] for row in eligible]: raise ValueError('Paired identities/order changed')
    rng=np.random.default_rng(freeze['uncertainty']['seed']);replicates=freeze['uncertainty']['replicates'];boot={};contrasts=[]
    for cat in categories:
        n=len(values[cat][profiles[0]]);draws=rng.integers(0,n,size=(replicates,n));boot[cat]={}
        for left,right in pairs:
            diff=values[cat][left]-values[cat][right];sample=diff[draws].mean(1);boot[cat][left,right]=sample
            lo,hi=np.quantile(sample,[.025,.975]);contrasts.append({'category':cat,'left':left,'right':right,'difference_pp':float(diff.mean()*100),
                'lower95_pp':float(lo*100),'upper95_pp':float(hi*100),'n_anomalous_units':n})
    for left,right in pairs:
        sample=np.mean([boot[cat][left,right] for cat in categories],axis=0);lo,hi=np.quantile(sample,[.025,.975])
        point=np.mean([(values[cat][left]-values[cat][right]).mean() for cat in categories])
        contrasts.append({'category':'equal_two_category_macro','left':left,'right':right,'difference_pp':float(point*100),
            'lower95_pp':float(lo*100),'upper95_pp':float(hi*100),'n_anomalous_units':sum(len(values[cat][left]) for cat in categories)})
    # Compare only endpoints whose source data are actually in PER_UNIT.csv.
    if (evidence/'SUMMARY.csv').exists():
        with (evidence/'SUMMARY.csv').open(encoding='utf-8-sig',newline='') as stream: expected=list(csv.DictReader(stream))
        for got in summaries:
            old=next(row for row in expected if row['category']==got['category'] and row['profile']==got['profile'])
            for key in ('Unit_AP','Image_AP','Image_AUROC','sensitivity','FAR','coverage_0.01','coverage_0.05','gated_coverage_0.01','gated_coverage_0.05'):
                if abs(got[key]-float(old[key]))>1e-12: raise ValueError('Scalar replay differs: '+repr((got['category'],got['profile'],key)))
    if (evidence/'CONTRASTS.csv').exists():
        with (evidence/'CONTRASTS.csv').open(encoding='utf-8-sig',newline='') as stream: expected=list(csv.DictReader(stream))
        for got,old in zip(contrasts,expected,strict=True):
            if (got['category'],got['left'],got['right'])!=(old['category'],old['left'],old['right']): raise ValueError('Contrast identity/order changed')
            for key in ('difference_pp','lower95_pp','upper95_pp'):
                if abs(got[key]-float(old[key]))>1e-12: raise ValueError('Contrast replay differs')
    out.mkdir(parents=True);table(out/'RECOMPUTED_UNIT_AP_AND_IMAGE_METRICS.csv',summaries);table(out/'RECOMPUTED_CONTRASTS.csv',contrasts)
    record={'status':'PASS','scope':'Scalar Unit AP/native image/coverage/alarms and paired pointwise intervals; pooled pixel AP/AUROC are not scalar-reconstructable.',
            'source_PER_UNIT_sha256':hashlib.sha256((evidence/'PER_UNIT.csv').read_bytes()).hexdigest(),
            'source_FROZEN_sha256':hashlib.sha256((evidence/'FROZEN.json').read_bytes()).hexdigest(),'GPU_calls':0,'image_or_GT_access':False}
    (out/'REPLAY_COMPLETE.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');print(json.dumps(record))
if __name__=='__main__':main()
