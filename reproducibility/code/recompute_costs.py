"""Recompute reported latency statistics from individual actual observations."""
from pathlib import Path
import argparse,json,csv,math,statistics,hashlib
import numpy as np
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--records',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists(),'Use a new output directory';a.output.mkdir(parents=True)
    records=json.loads(a.records.read_text('utf-8-sig'))['records'];rows=[]
    for item in records:
        d=item['observation'];warm=np.array([x['total_request_s'] for x in d['samples'] if x['phase']=='resident_warm'],np.float64)
        assert len(warm)==d['resident_warm_n'] and len(warm)>0 and np.isfinite(warm).all() and (warm>0).all()
        vals=dict(resident_warm_mean_s=float(warm.mean()),resident_warm_median_s=float(np.median(warm)),resident_warm_p95_s=float(np.quantile(warm,.95)))
        for k,v in vals.items():assert math.isclose(v,d[k],rel_tol=1e-12,abs_tol=1e-12),(item['category'],item['method'],k)
        first=[x for x in d['samples'] if x['phase']=='first'];assert len(first)==1 and first[0]['total_request_s']==d['first_request_s']
        rows.append(dict(cohort=item['cohort'],category=item['category'],method=item['method'],first_request_s=d['first_request_s'],resident_warm_n=len(warm),**vals,original_observation_sha256=item['original_observation_sha256']))
    with (a.output/'COST_SUMMARY.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (a.output/'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE_ACTUAL_OBSERVATION_STATISTICS',records=len(rows),input_sha256=sha(a.records),summary_sha256=sha(a.output/'COST_SUMMARY.csv'),scope='Individual observed raw requests supply all latency statistics. This command does not perform inference, refit normal models or measure another device.'),indent=2)+'\n',encoding='utf-8')
    print('Recomputed',len(rows),'measured configuration summaries')
if __name__=='__main__':main()
