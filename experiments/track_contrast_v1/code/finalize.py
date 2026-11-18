"""Wait for all fixed-endpoint pipelines, then perform only registered analyses."""
import json
import subprocess
import sys
import time
from runtime import E,save_json


def main():
    receipt=E/'runtime/POSTPROCESS.json'
    if receipt.exists():raise FileExistsError(receipt)
    record={'status':'waiting_for_all_fixed_endpoints','phases':[]};save_json(receipt,record)
    while True:
        states=[json.loads((E/'runtime'/f'formal_{c}.json').read_text())['status'] for c in 'ABC']
        if any(s=='FAILED' for s in states):raise RuntimeError('A formal pipeline failed')
        if all(s=='COMPLETE' for s in states):break
        time.sleep(5)
    record['status']='running';save_json(receipt,record)
    for args in [['evaluate.py','--summary'],['validate_final.py'],['compute_report.py'],['report_tables.py']]:
        row={'args':args,'start_unix':time.time()};record['phases'].append(row);save_json(receipt,record)
        result=subprocess.run([sys.executable,str(E/'code'/args[0]),*args[1:]])
        row.update(returncode=result.returncode,end_unix=time.time());save_json(receipt,record)
        if result.returncode:
            record['status']='failed';save_json(receipt,record);raise SystemExit(result.returncode)
    record.update(status='complete',end_unix=time.time());save_json(receipt,record)

if __name__=='__main__':main()
