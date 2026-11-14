"""Synthetic saved request integrity and summary calculation checks."""
from pathlib import Path
import json, tempfile, unittest, uuid, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
import request_statistics as a

class RequestStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='aggregation_fixture_',dir=None)
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.group=dict(reference=[dict(global_identity='fixture:fit')],
            calibration=[dict(global_identity=f'fixture:cal:{i}',real_view_boxes=[[0,0,1,1]]) for i in range(64)],
            query=[dict(global_identity='fixture:normal',real_view_boxes=[[0,0,1,1]]),
                   dict(global_identity='fixture:bad',real_view_boxes=[[0,0,1,1]])])
        self.config=dict(config_id='a'*64,state_id='b'*64,cohort_id='fixture_seen',category='candle',
            profile_id='M_W_U',budget_id='B',expected_counts=dict(reference=1,calibration=64,query=2))

    def timing_fixture(self):
        folder=self.root/('fixture_timing_'+uuid.uuid4().hex);folder.mkdir()
        schedule=[('first',0),('warmup',1),('warmup',2)]+[('resident',i%3) for i in range(27)]
        samples=[dict(request_index=i,phase=phase,unit_id=self.group['calibration'][index]['global_identity'],
            state_sha256='state_fixture',process_id=12345,actual_views=1,total_request_s=.1,
            encoder_readings_per_view=4,distance_queries_per_anchor=1,actual_encoder_calls=4,actual_PCA_calls=0,
            scope='SYNTHETIC_TIMING_FIXTURE_ONLY') for i,(phase,index) in enumerate(schedule)]
        env=dict(platform='win32',scope='SYNTHETIC_FIXTURE_METADATA_NOT_REAL_ENVIRONMENT')
        (folder/'ENVIRONMENT.json').write_text(json.dumps(env),encoding='utf-8')
        summary=dict(category='candle',profile_id='M_W_U',budget_id='B',state_sha256='state_fixture',process_id=12345,
            resident_n=27,resident_median_s=.1,resident_p95_s=.1,environment_sha256=a.sha(folder/'ENVIRONMENT.json'),
            source_kind='RAW_RECOMPUTED',single_profile_process=True,query_feature_cache=False,query_mask_cache=False,
            input_SHA_inside_timer=False,RGB_decode_inside_timer=True,output_validation_inside_timer=False,persistence_inside_timer=False)
        request=dict(config_id=self.config['config_id'],state_sha256='state_fixture',samples=samples)
        (folder/'REQUESTS.json').write_text(json.dumps(request),encoding='utf-8')
        (folder/'TIMING_SUMMARY.json').write_text(json.dumps(summary),encoding='utf-8')
        self.seal_timing_fixture(folder)
        return folder

    def seal_timing_fixture(self,folder):
        complete=dict(status='COMPLETE_TIMING',requests_sha256=a.sha(folder/'REQUESTS.json'),summary_sha256=a.sha(folder/'TIMING_SUMMARY.json'))
        (folder/'COMPLETE.json').write_text(json.dumps(complete),encoding='utf-8')

    def test_exact_30_request_schedule_and_quantiles_validate(self):
        folder=self.timing_fixture()
        result=a.read_timing(folder,self.config,'state_fixture',self.group)
        self.assertEqual(len(result['requests']),30)
        self.assertEqual(result['summary']['resident_n'],27)

    def test_wrong_request_encoder_count_refused(self):
        folder=self.timing_fixture();doc=a.read(folder/'REQUESTS.json');doc['samples'][2]['actual_encoder_calls']=1
        (folder/'REQUESTS.json').write_text(json.dumps(doc),encoding='utf-8');self.seal_timing_fixture(folder)
        with self.assertRaisesRegex(ValueError,'schedule mismatch'):
            a.read_timing(folder,self.config,'state_fixture',self.group)

    def test_request_hash_and_quantile_tamper_refused(self):
        folder=self.timing_fixture();doc=a.read(folder/'TIMING_SUMMARY.json');doc['resident_p95_s']=.001
        (folder/'TIMING_SUMMARY.json').write_text(json.dumps(doc),encoding='utf-8');self.seal_timing_fixture(folder)
        with self.assertRaisesRegex(ValueError,'p95 differs'):
            a.read_timing(folder,self.config,'state_fixture',self.group)

    def test_other_OS_timing_cannot_enter_Windows_series(self):
        folder=self.timing_fixture();env=a.read(folder/'ENVIRONMENT.json');env['platform']='linux'
        (folder/'ENVIRONMENT.json').write_text(json.dumps(env),encoding='utf-8')
        summary=a.read(folder/'TIMING_SUMMARY.json');summary['environment_sha256']=a.sha(folder/'ENVIRONMENT.json')
        (folder/'TIMING_SUMMARY.json').write_text(json.dumps(summary),encoding='utf-8');self.seal_timing_fixture(folder)
        with self.assertRaisesRegex(ValueError,'another OS'):
            a.read_timing(folder,self.config,'state_fixture',self.group)

