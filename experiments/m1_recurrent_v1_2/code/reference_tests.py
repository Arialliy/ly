"""Run the supplied tests unchanged in the existing pinned environment."""
from pathlib import Path
import json
import sys
import xml.etree.ElementTree as ET
import pytest

E = Path(__file__).resolve().parents[1]
reference = E / 'reference/M1_v12_state_update_reference'
sys.path.insert(0, str(reference))
output = E / 'checks/REFERENCE_TESTS.xml'
if output.exists():
    raise FileExistsError(output)
exit_code = pytest.main(['-q', str(reference/'test_state_update.py'),
                         '--junitxml='+str(output), '-p', 'no:cacheprovider'])
suites = list(ET.parse(output).getroot().iter('testsuite'))
result = {'exit_code': int(exit_code), 'status': 'PASS' if exit_code == 0 else 'FAIL',
          'tests': sum(int(s.attrib['tests']) for s in suites),
          'failures': sum(int(s.attrib['failures']) for s in suites),
          'errors': sum(int(s.attrib['errors']) for s in suites),
          'source': str(reference/'test_state_update.py'),
          'scope': 'Supplied CPU core tests; no detector performance claim'}
(E/'checks/REFERENCE_TESTS.json').write_text(json.dumps(result, indent=2)+'\n')
raise SystemExit(exit_code)
