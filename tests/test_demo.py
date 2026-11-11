import copy,json,subprocess,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from demo import load,fixture,inputs,merge_payloads,offline
class DemoTests(unittest.TestCase):
 def test_same_entity_keeps_different_assertions_and_scopes(self):
  r=offline();self.assertEqual(r['assertions'],4);self.assertEqual(r['distinct_entities'],1);self.assertTrue(r['single_vs_partitioned_payload_equal']);self.assertEqual({x['scope'] for x in r['evidence'] if x['attribute']=='revenue'},{'segment Alpha','segment Beta'});self.assertFalse(r['latency_measured'])
 def test_repeated_exact_assertion_deduplicates_without_entity_flattening(self):
  rows=[x['payload'] for x in fixture()];self.assertEqual(len(merge_payloads(rows+rows)),4)
 def test_conflict_in_same_assertion_build_is_not_silently_overwritten(self):
  rows=[x['payload'] for x in fixture()];changed=json.loads(rows[0]);changed['value']='999'
  with self.assertRaises(ValueError):merge_payloads(rows+[json.dumps(changed)])
 def test_versioned_assertion_does_not_overwrite_prior_build(self):
  row=fixture()[0]['payload'];changed=json.loads(row);changed['build_id']='synthetic-v2';self.assertEqual(len(merge_payloads([row,json.dumps(changed)])),2)
 def test_default_cli_never_claims_database_or_model_execution(self):
  p=subprocess.run([sys.executable,str(ROOT/'demo.py')],capture_output=True,text=True,check=True);d=json.loads(p.stdout);self.assertEqual((d['model_calls'],d['database_calls']),(0,0));self.assertTrue(d['synthetic_fixture'])
 def test_out_of_budget_request_is_rejected(self):
  for count in [0,33]:
   with self.assertRaises(ValueError):inputs('parallel',4,count)
 def test_loader_stops_before_any_mutation_on_a_nonempty_database(self):
  calls=[]
  class Result:
   def single(self):return {'n':1}
  class Session:
   def __enter__(self):return self
   def __exit__(self,*args):pass
   def run(self,q):calls.append(q);return Result()
  class Driver:
   def session(self,**kwargs):return Session()
  with self.assertRaises(ValueError):load(Driver())
  self.assertEqual(calls,['MATCH (n) RETURN count(n) AS n'])
if __name__=='__main__':unittest.main()
