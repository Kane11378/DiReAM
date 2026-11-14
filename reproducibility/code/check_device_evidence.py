"""Validate device-check evidence without loading a model or using CUDA."""
from pathlib import Path
import argparse, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_bindings as portable
from package_integrity import verify

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,required=True,help='Current raw-rebuild validation bundle')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    verify(root)
    from public_rebuild_evidence import validate_bundle
    result=validate_bundle(root,args.evidence.resolve())
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        print('device checks: '+str(error), file=sys.stderr)
        raise SystemExit(2)
