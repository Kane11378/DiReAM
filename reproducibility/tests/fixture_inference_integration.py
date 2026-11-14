"""Explicit synthetic cross-module CLI fixture; never a scientific entry point.

Runs the actual runner atomic prediction/seal code, original evaluator numerical
core and original whole-unit bootstrap. Encoder/decoded RGB/GT are generated
test fixtures. It asserts that production evaluation refuses fixture inputs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

RUN=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RUN/'code'))
import inference_runner as runner
import evaluation as evaluator
from state_store import atomic_json,file_sha,member_digest,read_json


def unit(identity,label):
    return dict(global_identity=identity,label_stratum=label,
                source_payload_sha256=hashlib.sha256(identity.encode()).hexdigest(),
                real_view_boxes=[[0,0,256,256],[256,0,512,256]],composite_wh=[512,256])


class SyntheticRGB:
    @staticmethod
    def decode_unit(row,context,verify=True):
        assert verify is True
        marker=255 if row['label_stratum']=='anomaly' else 0
        return [np.full((1,1,3),marker,np.uint8) for _ in row['real_view_boxes']]


class SyntheticEncoder:
    def __init__(self,method,fail_at=None):
        self.method=method;self.calls=0;self.fail_at=fail_at
    def predict_views(self,views):
        self.calls+=1
        if self.calls==self.fail_at:
            raise RuntimeError('explicit synthetic checkpoint interruption')
        maps=np.zeros((len(views),256,256),np.float32)
        abnormal=int(views[0][0,0,0])==255
        score=.1
        if abnormal:
            score=.9 if self.method=='M_W_U' else .8
            coordinate=(0,0) if self.method=='M_W_U' else (1,1)
            maps[:,coordinate[0],coordinate[1]]=score
        return dict(maps256=maps,object_score=np.float64(score))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    output=args.output.resolve()
    if output.is_relative_to(RUN.resolve()) or 'fixture' not in output.name.lower():
        raise ValueError('Synthetic fixture output must be outside the reader and named fixture')
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    context=dict(run_root=str(output),synthetic_fixture=True)
    group=dict(reference=[unit('fixture:fit:0','normal')],
        calibration=[unit(f'fixture:cal:{i}','normal') for i in range(64)],
        query=[unit('fixture:q:normal','normal'),unit('fixture:q:bad','anomaly')])
    gt=np.zeros((2,2,256,256),bool);gt[1,:,0,0]=True
    bad=np.array([False,True])
    stats=evaluator._pinned_module(evaluator.SOURCE_DEFAULTS,'statistics_core')
    evaluation_dirs=[];predictions=[];production_refusals=0
    for method in ['M_W_U','I_W_U']:
        config=dict(config_id='synthetic_crossmodule_'+method,category='candle',profile_id=method,
                    budget_id='B',cohort_id='visa12',state_sha256=hashlib.sha256(method.encode()).hexdigest(),
                    membership_digest=member_digest(group))
        pred_dir=output/(method+'_predictions')
        interrupted=False
        try:
            runner.predict_with_runtime(config,context,SyntheticEncoder(method,3),group,pred_dir,SyntheticRGB,
                                        source_kind='RAW_RECOMPUTED')
        except RuntimeError as exc:
            assert 'synthetic checkpoint' in str(exc)
            interrupted=True
        assert interrupted and not (pred_dir/'COMPLETE.json').exists()
        runtime=SyntheticEncoder(method)
        seal=runner.predict_with_runtime(config,context,runtime,group,pred_dir,SyntheticRGB,source_kind='RAW_RECOMPUTED')
        assert len(seal['records'])==66 and runtime.calls==64
        seal['synthetic_fixture']=True
        seal['source_kind_explanation']='Generated fake encoder outputs; RAW_RECOMPUTED is schema compatibility only; never production'
        atomic_json(pred_dir/'COMPLETE.json',seal,replace=True)
        checked,loaded=evaluator.validate_prediction_complete(pred_dir,group,config)
        try:
            evaluator.evaluate_predictions(pred_dir,group,config,evaluator.SOURCE_DEFAULTS,{},
                                            output/'forbidden_production',synthetic_fixture=True)
        except ValueError:
            production_refusals+=1
        else:raise AssertionError('Synthetic fixture entered production evaluation')
        summary,units=evaluator._compute_evaluation(group,config,checked,loaded,gt,bad,stats.binary_metrics)
        assert len(units)==2 and units[0]['object_pixel_AP'] is None and units[1]['actual_views']==2
        summary['source_kind']='SYNTHETIC_FIXTURE_ONLY'
        for item in units:item['source_kind']='SYNTHETIC_FIXTURE_ONLY'
        ev_dir=output/(method+'_evaluation');ev_dir.mkdir()
        evaluator.csv_new(ev_dir/'SUMMARY.csv',[summary]);evaluator.csv_new(ev_dir/'PER_UNIT.csv',units)
        evaluator.write_new(ev_dir/'COMPLETE.json',dict(status='SYNTHETIC_EVALUATION_COMPLETE',
            config=config,membership_digest=config['membership_digest'],
            outputs=[dict(file=p.name,sha256=file_sha(p)) for p in sorted(ev_dir.glob('*.csv'))]))
        evaluation_dirs.append(ev_dir)
        predictions.append(dict(profile=method,full_predictions=66,resumed_prediction_calls=runtime.calls,
                                complete_sha256=file_sha(pred_dir/'COMPLETE.json')))
    requirements=[dict(cohort_id='visa12',stratum='category',category_or_macro='candle',
        left_profile='M_W_U',right_profile='I_W_U',left_budget='B',right_budget='B',requirement_id='fixture_1')]
    bootstrap=evaluator._bootstrap_impl(evaluation_dirs,requirements,evaluator.SOURCE_DEFAULTS,
                                        output/'synthetic_statistics',True)
    report=dict(status='SYNTHETIC_CROSSMODULE_E2E_PASSED_NOT_SCIENCE',
        production_module_files=[dict(path=str(path),sha256=file_sha(path)) for path in
                                 [RUN/'code/inference_runner.py',RUN/'code/state_store.py',RUN/'code/evaluation.py']],
        calibration_units=64,query_units=2,views_per_unit=2,original_GT_opened=False,
        GPU_used=False,pretrained_model_used=False,production_fixture_refusals=production_refusals,
        predictions=predictions,bootstrap_n=10000,bootstrap_seed=20261022,bootstrap=bootstrap,
        validation='Actual runner resume/seal -> evaluator seal loader -> unchanged evaluator numerical core -> unchanged whole-unit bootstrap')
    atomic_json(output/'FIXTURE_COMPLETE.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
