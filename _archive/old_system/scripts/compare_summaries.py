from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
CORE=['route_exact_match','route_f1','evidence_recall','evidence_precision','evidence_density','abstention_correctness','num_outdated','latency_seconds']
def main():
    p=argparse.ArgumentParser(); p.add_argument('--before',required=True); p.add_argument('--after',required=True); p.add_argument('--out',default=None); a=p.parse_args(); before=pd.read_csv(a.before); after=pd.read_csv(a.after); m=before.merge(after,on='method',suffixes=('_before','_after')); available=[]
    for metric in CORE:
        b,f=f'{metric}_before',f'{metric}_after'
        if b in m and f in m: m[f'{metric}_delta']=m[f]-m[b]; available.append(metric)
    out=Path(a.out) if a.out else Path(a.after).parent/'summary_comparison.csv'; out.parent.mkdir(parents=True,exist_ok=True); m.to_csv(out,index=False); print(m[['method']+[f'{x}_delta' for x in available]].to_string(index=False)); print(f'\nSaved comparison to {out}')
if __name__=='__main__': main()
