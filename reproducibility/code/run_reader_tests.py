"""Run software checks using generated arrays in an explicit scratch directory.

The core suite checks storage, evaluation, reuse and request aggregation.
The operator suite also checks exact-K selection and directional operations.
Neither suite loads pretrained models or runs image-dataset inference.
"""
from pathlib import Path
import argparse,importlib.metadata,json,sys,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scratch',type=Path,required=True);p.add_argument('--suite',choices=['core','operators'],default='core');a=p.parse_args()
    a.scratch.mkdir(parents=True,exist_ok=True);tempfile.tempdir=str(a.scratch.resolve())
    loader=unittest.TestLoader();suite=unittest.TestSuite()
    if a.suite=='core':
        for name in ['test_inference_runner.py','test_evaluation.py','test_state_reuse.py','test_request_statistics.py']:
            suite.addTests(loader.discover(str(ROOT/'tests'),pattern=name,top_level_dir=str(ROOT/'tests')))
    else:
        for folder in ['feature_adapter','directional_adapter']:
            suite.addTests(loader.discover(str(ROOT/'tests'/folder),pattern='test*.py',top_level_dir=str(ROOT/'tests'/folder)))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    versions={}
    for package in ['numpy','Pillow','scipy','torch','tqdm','opencv-python-headless','scikit-learn']:
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]='NOT_INSTALLED_IN_THIS_ROUTE_ENVIRONMENT'
    print(json.dumps(dict(tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),suite=a.suite,synthetic_only=True,python=sys.version,packages=versions)))
    return 0 if result.wasSuccessful() else 1
if __name__=='__main__':raise SystemExit(main())
