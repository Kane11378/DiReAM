"""Recompute saved observations: verify, macros, costs, budgets or export.

Use --help for the complete command list and --output for a new results directory.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from package_integrity import verify
import saved_results

if __name__ == '__main__':
    try:
        if not any(x in sys.argv[1:] for x in ('-h','--help')):
            verify(Path(__file__).resolve().parents[1])
        saved_results.main()
    except (ValueError, OSError, KeyError) as error:
        print('reader: '+str(error), file=sys.stderr)
        raise SystemExit(2)
