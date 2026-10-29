#!/usr/bin/env python3
"""Read-only evaluation of human-gold applicability for V0-V3; no fake numbers.

Usage:
  python qa/evaluate_applicability.py --gold data/ANNOTATION_GOLD.csv --pred results/RAW_PREDICTIONS.csv --out results/eval

Important: gold rows are independent real annotations. Synthetics cannot be substituted.
"""
import argparse,csv,math,json
from collections import defaultdict
from pathlib import Path
VALID={'DIRECT','ANALOG','OUT_OF_SCOPE','UNRESOLVED'}

def rows(path):
 with Path(path).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def wilson(success,n,z=1.95996398454):
 if not n:return [None,None]
 p=success/n;d=1+z*z/n;center=(p+z*z/(2*n))/d
 v=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return [max(0.0,center-v),min(1.0,center+v)]

def evaluate(gold,pred,split="TEST"):
 if not gold or not pred:raise ValueError('Cannot score empty gold/pred; no synthetic fallback')
 if split not in ("TEST","DEV"):raise ValueError("Only DEV and TEST permitted")
 families={}
 for r in gold:
  k=r.get("source_family", "").strip()
  s=r.get("split", "").strip().upper()
  if s not in ("TEST","DEV"):raise ValueError("Only human-labeled DEV/TEST, no SYNTHETIC")
  if k in families and families[k]!=s:raise ValueError("Source family leaks between DEV and TEST: "+k)
  families[k]=s
 gold=[r for r in gold if r.get("split", "").strip().upper()==split]
 if not gold:raise ValueError("No examples in selected split "+split)
 G={}
 for r in gold:
  ident=r.get('sample_id','').strip()
  if not ident:raise ValueError('Missing sample_id in gold')
  if ident in G:raise ValueError('Duplicate gold sample_id '+ident)
  status=r.get('gold_applicability','').strip()
  if status not in VALID:raise ValueError('Gold status invalid: '+status)
  if not r.get('query_id','').strip() or not r.get('source_family','').strip():raise ValueError('Missing query_id/source_family')
  if (r.get('split') or '').strip().upper() not in ('TEST','DEV'):
   raise ValueError('Use independent gold labeled TEST/DEV only, never SYNTHETIC')
  G[ident]=r
 B=defaultdict(list)
 seen=set()
 for r in pred:
  key=(r.get('baseline',''),r.get('sample_id',''))
  if key in seen:raise ValueError('Duplicate prediction '+str(key))
  seen.add(key)
  if key[1] not in G:raise ValueError('Prediction sample not in gold: '+key[1])
  if r.get('predicted_status') not in VALID:raise ValueError('Prediction status invalid: '+str(r.get('predicted_status')))
  if r.get('query_id','')!=G[key[1]]['query_id']:raise ValueError('Mismatched query_id for '+key[1])
  B[key[0]].append((G[key[1]],r))
 out={}
 for name,pairs in B.items():
  if not name:raise ValueError('Empty baseline name')
  if len(pairs)!=len(G):raise ValueError('Baseline '+name+' has '+str(len(pairs))+' of '+str(len(G))+' samples; refuse comparison')
  foreign=[(g,p) for g,p in pairs if g['gold_applicability'] in ('ANALOG','OUT_OF_SCOPE')]
  false_foreign=sum(p['predicted_status']=='DIRECT' for g,p in foreign)
  admitted=[(g,p) for g,p in pairs if p['predicted_status']=='DIRECT']
  false_admitted=sum(g['gold_applicability']!='DIRECT' for g,p in admitted)
  answerable={g['query_id'] for g,p in pairs if g['gold_applicability']=='DIRECT'}
  correct_q={g['query_id'] for g,p in pairs if g['gold_applicability']=='DIRECT' and p['predicted_status']=='DIRECT'}
  qs_with_false={g['query_id'] for g,p in pairs if p['predicted_status']=='DIRECT' and g['gold_applicability']!='DIRECT'}
  safe_q=correct_q-qs_with_false
  conflict=sum(g['gold_applicability']!=p['predicted_status'] for g,p in pairs)
  out[name]={
   'n_samples':len(pairs),'n_document_families':len({g['source_family'] for g,p in pairs}),
   'n_foreign_scope':len(foreign),'false_direct_foreign':false_foreign,
   'false_transfer_rate':false_foreign/len(foreign) if foreign else None,
   'false_transfer_ci_wilson95':wilson(false_foreign,len(foreign)),
   'n_answerable_questions':len(answerable),'n_queries_with_correct_direct':len(correct_q),
   'direct_coverage':len(correct_q)/len(answerable) if answerable else None,
   'direct_coverage_ci_wilson95':wilson(len(correct_q),len(answerable)),
   'safe_direct_coverage':len(safe_q)/len(answerable) if answerable else None,
   'n_accepted_direct':len(admitted),'n_false_accepted_direct':false_admitted,
   'selective_risk':false_admitted/len(admitted) if admitted else None,
   'abstention_rate':sum(p['predicted_status']=='UNRESOLVED' for g,p in pairs)/len(pairs),
   'overall_disagreement_rate':conflict/len(pairs),
   'note':'All counts depend on independent gold; no significance claim without paired cluster CI'
  }
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gold',required=True);ap.add_argument('--pred',required=True);ap.add_argument('--out',required=True);ap.add_argument('--split',choices=["DEV","TEST"],default="TEST")
 a=ap.parse_args();g=rows(a.gold);p=rows(a.pred);r=evaluate(g,p,split=a.split)
 out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
 out.with_suffix('.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 with out.with_suffix('.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.writer(f);w.writerow(['baseline','n_samples','n_foreign_scope','false_direct_foreign','false_transfer_rate','n_answerable_questions','direct_coverage','safe_direct_coverage','selective_risk','abstention_rate'])
  for k,v in r.items():w.writerow([k,*[v.get(x) for x in ['n_samples','n_foreign_scope','false_direct_foreign','false_transfer_rate','n_answerable_questions','direct_coverage','safe_direct_coverage','selective_risk','abstention_rate']]])
 print(json.dumps(r,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
