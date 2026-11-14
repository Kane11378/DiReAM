"""Inspect or run the fixed image-inference configurations.

plan and check-resources do not construct a model. See docs/INFERENCE.md.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from package_integrity import verify
import image_inference

if __name__ == '__main__':
    try:
        if not any(x in sys.argv[1:] for x in ('-h','--help')):
            verify(Path(__file__).resolve().parents[1])
        raise SystemExit(image_inference.main())
    except (ValueError, OSError, KeyError) as error:
        print('inference: '+str(error), file=sys.stderr)
        raise SystemExit(2)
