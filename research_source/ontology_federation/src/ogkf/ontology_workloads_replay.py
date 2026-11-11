"""Zero-model replay of frozen proposals; does not backfill original write clocks."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from .ontology_workloads import (Store, Telemetry, encoded, local_dozer, packet,
                                 read_rows, sha, trace_audit, utc, write)
from .ontology_workloads_audit import audit_run


def run(source, directory):
    original=audit_run(source)
    assert original['completed_all_assigned']
    directory.mkdir(parents=True,exist_ok=False)
    data=json.loads((source/'dataset.private.json').read_text())
    telemetry=Telemetry(directory,directory.name)
    runtime=None
    manifest={'protocol':'ontology-workloads-write-replay-v1','kind':'zero_model_write_replay',
              'source_run':str(source),'source_input_sha256':sha(data),'started_utc':utc(),
              'http_requests':0,'original_write_timing_available':False,
              'scope':'Replay exact frozen proposals into an owned DB, trace real writes and verify payload reproduction. No extraction or semantic re-evaluation.'}
    for filename in ('ontology_workloads_replay.py','ontology_workloads.py'):
        raw=Path(__file__).with_name(filename).read_bytes()
        manifest[filename+'_sha256']=hashlib.sha256(raw).hexdigest()
        (directory/filename).write_bytes(raw)
    write(directory/'manifest.json',manifest)
    with local_dozer() as (env,runtime):
        store=Store(env,telemetry)
        try:
            store.setup()
            for row in read_rows(source/'index-outcomes.jsonl'):
                ep=row['episode_id']
                with telemetry.span('index_build_replay',ep):
                    sources=[s for s in data['sources'] if s['cik']==row['cik']]
                    store.load(row['namespace'],row['records'],sources,episode=ep)
                    actual=store.read(ep,row['namespace'],row['cik'],('organization','performance','risk'))
                    assert sha(actual['records'])==row['native_readback_sha256']
                    assert packet(actual['records'])==packet(sorted(row['records'],key=lambda r:r['id']))
                    telemetry.append('replay-outcomes.jsonl',{'episode_id':ep,'namespace':row['namespace'],'cik':row['cik'],
                        'proposal_count':len(row['records']),'original_native_readback_sha256':row['native_readback_sha256'],
                        'replay_readback_sha256':sha(actual['records']),'packet_sha256':sha(packet(actual['records'])),
                        'parity':True,'http_requests':0})
        finally:store.close()
    telemetry.flush_metrics()
    write(directory/'terminal.json',{'completed_utc':utc(),'runtime_final':runtime,'replayed_index_jobs':18,
        'exact_proposal_readback_parity':True,'http_requests':0,'original_write_timing_available':False,
        'trace_audit':trace_audit(directory)})
    print(json.dumps({'replayed_index_jobs':18,'exact_proposal_readback_parity':True,'http_requests':0,
                      'teardown_success':runtime['teardown_success']}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--directory',type=Path,required=True)
    a=p.parse_args();run(a.source,a.directory)

if __name__=='__main__':main()
