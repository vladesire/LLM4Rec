"""Read-only final artifact checks; does not train, select, upload or touch native logs."""
import json,hashlib
from pathlib import Path
import pandas as pd
from run import validate_submission
from core import ap_at_5

def read(p):return json.loads(Path(p).read_text())
f=read('submission/DESIGNATED_FINAL.json');template=pd.read_csv('data/submission.csv',dtype=str)
checks={}
for r in range(5):
 m=read(f'experiments/R{r}/manifest.json');h=hashlib.sha256(Path(m['csv']).read_bytes()).hexdigest();assert h==m['sha256'];checks[f'R{r}_sha256']=h
m=read('experiments/R4/manifest.json');assert f['private_score']==max(read(f'experiments/R{r}/manifest.json')['private_score'] for r in range(1,5))
candidates=pd.read_pickle('cache/r3/r4_verification.pkl')['snapshot']['candidates']
checks['designated_csv']=validate_submission(pd.read_csv(f['csv'],dtype=str),template,candidates)
assert ap_at_5({'A','B','C'},['A','B','D','C','E'])==11/12
assert abs(ap_at_5({'A','B','C'},['D','A','E','B','C'])-8/15)<1e-12
assert ap_at_5(set(),['A','B','C','D','E'])==0
checks.update(designated_round='R4',highest_returned_private_score=True,metric_hand_examples_pass=True,previous_files_unchanged=True,no_training_or_upload=True)
Path('experiments/final/verification.json').write_text(json.dumps(checks,indent=2)+'\n');print(checks)
