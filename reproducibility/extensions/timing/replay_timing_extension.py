"""Recompute new supplemental request statistics from relative CSV records.

Pure Python standard library. No images, models, CUDA or workstation paths are
opened. These records are independent of the historical reader timing series.
"""
import argparse, csv, hashlib, json, math, statistics
from pathlib import Path

def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write_csv(path,rows):
    with path.open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def percentile(values,p):
    a=sorted(values);z=(len(a)-1)*p;i=math.floor(z);j=math.ceil(z)
    return a[i]*(j-z)+a[j]*(z-i) if i!=j else a[i]
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records',type=Path,default=Path(__file__).resolve().parent/'records')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();records=args.records.resolve();output=args.output.resolve()
    assert not output.exists(),'Choose a new output directory'
    manifest=json.loads((records/'MANIFEST.json').read_text('utf8'))
    for name,expected in manifest['files'].items():
        path=(records/name).resolve();assert path.is_relative_to(records)
        assert sha(path)==expected,(name,'changed record')
    protocol=json.loads((records/'PROTOCOL.json').read_text('utf8'))
    assert sha(records/'PROTOCOL.json')==json.loads((records/'PROTOCOL_HASH.json').read_text('utf8'))['sha256']
    requests=read_csv(records/'ALL_REQUESTS.csv');saved=read_csv(records/'PROCESS_SUMMARIES.csv')
    assert len(saved)==75 and len(requests)==3000
    rows=[]
    for job in protocol['jobs']:
        observations=[r for r in requests if r['job_id']==job['job_id']]
        resident=[r for r in observations if r['phase']=='resident']
        assert len(observations)==40 and len(resident)==36
        assert [int(r['input_index']) for r in observations if r['phase']=='first']==[protocol['first_request']]
        assert [int(r['input_index']) for r in observations if r['phase']=='warmup']==protocol['warmups']
        assert [int(r['input_index']) for r in resident]==job['resident_order']
        assert len({r['global_identity'] for r in resident})==18
        assert {r['input_role'] for r in resident}=={'calibration','query_normal','query_anomalous'}
        assert len({r['process_id'] for r in observations})==1
        for r in observations:
            assert r['category']==job['category'] and r['profile']==job['profile'] and int(r['block'])==job['block']
            selected_input=protocol['groups'][job['category']]['inputs'][int(r['input_index'])]
            assert r['global_identity']==selected_input['global_identity'] and r['input_role']==selected_input['role']
        expected_calls=1 if job['profile']=='N_W_F0' else 4
        expected_pca=0 if job['profile'].endswith('_U') else expected_calls
        assert all(int(r['encoder_calls'])==expected_calls and int(r['pca_calls'])==expected_pca for r in observations)
        values=[float(r['total_request_s'])*1000 for r in resident]
        original=next(r for r in saved if r['job_id']==job['job_id'])
        median=statistics.median(values);p95=percentile(values,.95)
        assert math.isclose(median,float(original['resident_median_ms']),abs_tol=1e-9)
        assert math.isclose(p95,float(original['resident_p95_ms']),abs_tol=1e-9)
        quality=protocol['groups'][job['category']]['quality'][job['profile']]
        assert math.isclose(float(original['unit_ap']),float(quality['unit_ap']),abs_tol=1e-15)
        assert original['historical_state_sha256']==quality['state_sha256']
        rows.append(dict(job_id=job['job_id'],category=job['category'],profile=job['profile'],block=job['block'],
            resident_n=36,distinct_inputs=18,resident_median_ms=median,resident_p95_ms=p95,
            unit_ap_percent=100*float(original['unit_ap']),historical_state_sha256=original['historical_state_sha256']))
    cells=[];contrasts=[]
    for category in protocol['categories']:
        for profile in protocol['profiles']:
            selected=[r for r in rows if r['category']==category and r['profile']==profile]
            values=[r['resident_median_ms'] for r in selected];tails=[r['resident_p95_ms'] for r in selected]
            assert len(selected)==3
            cells.append(dict(category=category,profile=profile,independent_processes=3,
                unit_ap_percent=selected[0]['unit_ap_percent'],process_medians_median_ms=statistics.median(values),
                process_medians_min_ms=min(values),process_medians_max_ms=max(values),
                process_p95s_median_ms=statistics.median(tails)))
        for mode in ['U','FC4']:
            reductions=[]
            for block in range(1,4):
                i=next(r for r in rows if r['category']==category and r['profile']=='I_W_'+mode and r['block']==block)
                m=next(r for r in rows if r['category']==category and r['profile']=='M_W_'+mode and r['block']==block)
                reductions.append(100*(1-m['resident_median_ms']/i['resident_median_ms']))
            ia=next(r['unit_ap_percent'] for r in cells if r['category']==category and r['profile']=='I_W_'+mode)
            ma=next(r['unit_ap_percent'] for r in cells if r['category']==category and r['profile']=='M_W_'+mode)
            contrasts.append(dict(category=category,mode=mode,unit_ap_mean_minus_independent_pp=ma-ia,
                reduction_median_percent=statistics.median(reductions),reduction_min_percent=min(reductions),reduction_max_percent=max(reductions),
                reduction_block1_percent=reductions[0],reduction_block2_percent=reductions[1],reduction_block3_percent=reductions[2]))
    output.mkdir(parents=True)
    write_csv(output/'PROCESSES_RECOMPUTED.csv',rows);write_csv(output/'CELLS_RECOMPUTED.csv',cells);write_csv(output/'CONTRASTS_RECOMPUTED.csv',contrasts)
    stage_check=None
    if (records/'stage_diagnostic/COMPLETE.json').exists():
        stage=records/'stage_diagnostic';stage_protocol=json.loads((stage/'PROTOCOL.json').read_text('utf8'))
        assert sha(stage/'PROTOCOL.json')==json.loads((stage/'PROTOCOL_HASH.json').read_text('utf8'))['sha256']
        stage_rows=[];stage_count=0
        for category in stage_protocol['categories']:
            for profile in stage_protocol['profiles']:
                folder=stage/'runs'/f'{category}_{profile}'
                obs=json.loads((folder/'REQUESTS.json').read_text('utf8'))
                selected=[r for r in obs if r['phase']=='resident_diagnostic'];assert len(selected)==9
                assert [r['input_index'] for r in selected]==stage_protocol['resident_inputs']
                assert len({r['global_identity'] for r in selected})==3
                total=statistics.mean(r['instrumented_total_request_s'] for r in selected)
                summary=json.loads((folder/'SUMMARY.json').read_text('utf8'))
                assert math.isclose(1000*total,summary['instrumented_mean_request_ms'],abs_tol=1e-9)
                cells_stage={k:statistics.mean(r[k] for r in selected) for k in selected[0] if k.endswith('_s') and k!='instrumented_total_request_s'}
                assert math.isclose(sum(cells_stage.values()),total,abs_tol=1e-9)
                for k,value in cells_stage.items():
                    assert math.isclose(1000*value,summary['stage_mean_ms'][k],abs_tol=1e-9)
                    assert math.isclose(100*value/total,summary['stage_mean_percent'][k],abs_tol=1e-9)
                stage_rows.append(dict(category=category,profile=profile,instrumented_mean_request_ms=1000*total,
                    **{k.replace('_s','_mean_ms'):1000*v for k,v in cells_stage.items()},
                    **{k.replace('_s','_percent'):100*v/total for k,v in cells_stage.items()}))
                stage_count+=len(selected)
        assert stage_count==72
        write_csv(output/'STAGES_RECOMPUTED.csv',stage_rows)
        stage_check=dict(processes=8,resident_diagnostic_requests=72,extra_synchronization=True,main_timing_not_pooled=True)
    audit=dict(status='PASSED',scope='New supplemental timing records only; historical data not pooled',
        processes=75,process_launches=75,distinct_os_process_ids=len({r['process_id'] for r in saved}),
        resident_requests=2700,total_requests=3000,distinct_inputs_per_category=18,
        source_protocol_sha256=sha(records/'PROTOCOL.json'),all_saved_process_statistics_reproduced=True,
        original_input_qualifiers=protocol['category_selection'],inference=False,fitting=False,GT_reads=0,
        separate_stage_diagnostic=stage_check)
    with (output/'REPLAY_CHECK.json').open('x',encoding='utf8') as f:json.dump(audit,f,indent=2)
    print(json.dumps(audit))

if __name__=='__main__':main()
