"""One GPU per condition: train fixed endpoint, infer both clips, then score."""
import argparse
import subprocess
import sys
import time
from runtime import E,SEQS,save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--condition',required=True,choices=['A','B','C'])
    condition=p.parse_args().condition
    receipt=E/'runtime'/f'PIPELINE_{condition}.json'
    if receipt.exists():raise FileExistsError(receipt)
    phases=[['train.py','--condition',condition]]
    for sequence in SEQS:
        phases.append(['infer.py','--condition',condition,'--sequence',sequence])
        phases.append(['evaluate.py','--condition',condition,'--sequence',sequence])
    record={'condition':condition,'status':'running','started_unix':time.time(),'phases':[]}
    save_json(receipt,record)
    for args in phases:
        row={'args':args,'start_unix':time.time()};record['phases'].append(row);save_json(receipt,record)
        process=subprocess.run([sys.executable,str(E/'code'/args[0]),*args[1:]])
        row.update(returncode=process.returncode,end_unix=time.time());save_json(receipt,record)
        if process.returncode:
            record.update(status='failed',end_unix=time.time());save_json(receipt,record)
            raise SystemExit(process.returncode)
    record.update(status='complete',end_unix=time.time());save_json(receipt,record)

if __name__=='__main__':main()
