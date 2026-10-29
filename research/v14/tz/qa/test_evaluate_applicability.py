import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from evaluate_applicability import evaluate, wilson

class ScorerTest(unittest.TestCase):
 def test_toy_no_leak(self):
  gold=[
   {'sample_id':'T1','query_id':'Q1','source_family':'A','gold_applicability':'DIRECT','split':'TEST'},
   {'sample_id':'T2','query_id':'Q2','source_family':'B','gold_applicability':'OUT_OF_SCOPE','split':'TEST'},
   {'sample_id':'T3','query_id':'Q3','source_family':'C','gold_applicability':'ANALOG','split':'TEST'}]
  pred=[{'sample_id':'T1','query_id':'Q1','baseline':'V3','predicted_status':'DIRECT'},{'sample_id':'T2','query_id':'Q2','baseline':'V3','predicted_status':'OUT_OF_SCOPE'},{'sample_id':'T3','query_id':'Q3','baseline':'V3','predicted_status':'DIRECT'}]
  r=evaluate(gold,pred)['V3']
  self.assertEqual(r['false_transfer_rate'],0.5)
  self.assertEqual(r['direct_coverage'],1.0)
  self.assertAlmostEqual(r['selective_risk'],0.5)
 def test_missing_prediction_rejected(self):
  g=[{'sample_id':'1','query_id':'q','source_family':'f','gold_applicability':'DIRECT','split':'TEST'},{'sample_id':'2','query_id':'q2','source_family':'f','gold_applicability':'DIRECT','split':'TEST'}]
  with self.assertRaises(ValueError):evaluate(g,[{'sample_id':'1','query_id':'q','baseline':'v','predicted_status':'DIRECT'}])
 def test_synthetic_rejected(self):
  with self.assertRaises(ValueError):evaluate([{'sample_id':'1','query_id':'q','source_family':'f','gold_applicability':'DIRECT','split':'SYNTHETIC'}],[{'sample_id':'1','query_id':'q','baseline':'v','predicted_status':'DIRECT'}])
 def test_wilson(self):
  low,hi=wilson(0,15);self.assertEqual(low,0);self.assertLess(hi,0.25)

if __name__=='__main__':unittest.main()
