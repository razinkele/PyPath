import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tests.test_detritus_consumption import test_deriv_includes_fish_discard_links

try:
    test_deriv_includes_fish_discard_links()
    print('TEST PASSED')
except AssertionError as e:
    print('TEST FAILED:', e)
except Exception:
    import traceback
    traceback.print_exc()
    print('TEST ERRORED')
