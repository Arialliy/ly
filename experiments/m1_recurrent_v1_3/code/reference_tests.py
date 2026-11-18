"""Run the supplied evidence-bank CPU tests without changing the reference."""
from pathlib import Path
import hashlib
import json
import sys
import xml.etree.ElementTree as ET

import pytest

E = Path(__file__).resolve().parents[1]
REFERENCE = E / 'reference/M1_v13_evidence_bank_reference'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output = E / 'checks/REFERENCE_TESTS.xml'
    receipt = E / 'checks/REFERENCE_TESTS.json'
    if output.exists() or receipt.exists():
        raise FileExistsError('Preserve existing reference-test evidence')
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = {str(p): sha(p) for p in (REFERENCE/'evidence_bank.py', REFERENCE/'test_evidence_bank.py')}
    sys.path.insert(0, str(REFERENCE))
    code = pytest.main(['-q', str(REFERENCE/'test_evidence_bank.py'),
                        '--junitxml='+str(output), '-p', 'no:cacheprovider'])
    suites = list(ET.parse(output).getroot().iter('testsuite'))
    counts = {key: sum(int(s.attrib.get(key, 0)) for s in suites)
              for key in ('tests', 'failures', 'errors', 'skipped')}
    unchanged = all(sha(path) == digest for path, digest in sources.items())
    passed = code == 0 and counts == {'tests':29, 'failures':0, 'errors':0, 'skipped':0} and unchanged
    result = {'status':'PASS' if passed else 'FAIL', 'pytest_exit_code':int(code), **counts,
              'source_sha256':sources, 'source_unchanged':unchanged, 'junit_sha256':sha(output),
              'scope':'Unmodified supplied CPU core tests with substitute operators; native DCN, detector training and full-frame costs are checked separately'}
    with receipt.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
