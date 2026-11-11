"""Export only this cohort, then verify actual Tempo span-ID readback."""
from __future__ import annotations
import argparse
import base64
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import urllib.request

from .ontology_workloads import read_rows, VERSION
from export_otel import (trace_payload, metric_payload, context_samples, post,
                         nanoseconds)


def span_ids(value):
    result=set()
    def scan(v):
        if isinstance(v,dict):
            for key,item in v.items():
                if key=='spanId' and isinstance(item,str):
                    if len(item)==16 and all(c in '0123456789abcdefABCDEF' for c in item):
                        result.add(item.lower())
                    else:
                        raw=base64.b64decode(item,validate=True)
                        if len(raw)!=8:raise ValueError('invalid Tempo span byte length')
                        result.add(raw.hex())
                else:scan(item)
        elif isinstance(v,list):
            for item in v:scan(item)
    scan(value)
    return result


def resource(payload, signal, run, protocol=VERSION):
    item=payload['resourceSpans' if signal=='traces' else 'resourceMetrics'][0]
    item['resource']['attributes'].extend([
        {'key':'service.instance.id','value':{'stringValue':run}},
        {'key':'experiment.protocol','value':{'stringValue':protocol}},
    ])
    return payload


def export(directory, *, readback=False):
    spans=read_rows(directory/'spans.jsonl');events=read_rows(directory/'events.jsonl')
    protocol=json.loads((directory/'manifest.json').read_text())['protocol']
    checkpoint=directory/'otel-exported.json'
    exported=set(json.loads(checkpoint.read_text())) if checkpoint.exists() else set()
    new=[s for s in spans if s['span_id'] not in exported]
    for offset in range(0,len(new),250):
        batch=new[offset:offset+250]
        post('http://127.0.0.1:19418','traces',resource(trace_payload(batch,events),'traces',directory.name,protocol))
        exported.update(s['span_id'] for s in batch)
        tmp=checkpoint.with_suffix('.tmp');tmp.write_text(json.dumps(sorted(exported))+'\n');tmp.replace(checkpoint)
    if spans:
        starts=[nanoseconds(e['utc']) for e in events if e.get('event')=='stage_started']
        metrics=metric_payload(spans,min(starts),time.time_ns(),context_samples(directory))
        post('http://127.0.0.1:19418','metrics',resource(metrics,'metrics',directory.name,protocol))
    result={'run':directory.name,'new_spans_exported':len(new),'total_spans':len(spans),'collector_acknowledged':len(exported)}
    if readback:
        expected=defaultdict(set)
        for span in spans:expected[span['trace_id']].add(span['span_id'][-16:])
        def fetch(item):
            trace,ids=item;seen=set();status=None
            for attempt in range(5):
                try:
                    with urllib.request.urlopen('http://127.0.0.1:19480/api/traces/'+trace,timeout=10) as response:
                        payload=json.load(response);status=response.status
                    seen.update(span_ids(payload))
                    if ids<=seen:break
                except Exception as e:status=type(e).__name__
                time.sleep(1)
            return {'trace_id':trace,'expected_spans':len(ids),'matched_spans':len(ids&seen),'missing_span_ids':sorted(ids-seen),'http_status':status}
        rows=[fetch(item) for item in expected.items()]
        result.update(trace_count=len(rows),fully_read_back=sum(not r['missing_span_ids'] for r in rows),
                      all_spans_read_back=all(not r['missing_span_ids'] for r in rows),traces=rows)
        (directory/'tempo-readback.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='traces'}),flush=True)
    return result


def repair_missing(directory):
    """Replay only confirmed missing canonical spans, retaining their IDs.

    This does not call the model or database. Collector ACK is not sink ACK.
    """
    previous=json.loads((directory/'tempo-readback.json').read_text())
    archive=directory/'tempo-readback-pre-recovery.json'
    if not archive.exists():archive.write_text(json.dumps(previous,indent=2)+'\n')
    spans=read_rows(directory/'spans.jsonl');events=read_rows(directory/'events.jsonl')
    expected=defaultdict(set)
    for span in spans:expected[span['trace_id']].add(span['span_id'][-16:])
    missing={}
    for row in previous['traces']:
        if not row['missing_span_ids']:continue
        seen=set()
        try:
            with urllib.request.urlopen('http://127.0.0.1:19480/api/traces/'+row['trace_id'],timeout=10) as response:
                seen=span_ids(json.load(response))
        except urllib.error.HTTPError as error:
            if error.code!=404:raise
        missing[row['trace_id']]=expected[row['trace_id']]-seen
    replay=[s for s in spans if s['span_id'][-16:] in missing.get(s['trace_id'],set())]
    for offset in range(0,len(replay),100):
        post('http://127.0.0.1:19418','traces',resource(trace_payload(replay[offset:offset+100],events),'traces',directory.name))
    receipt={'confirmed_missing_canonical_spans_reexported':len(replay),'trace_ids':sorted(k for k,v in missing.items() if v),
             'span_ids_preserved':True,'model_or_database_calls':0,'collector_ack_is_sink_ack':False}
    (directory/'canonical-span-repair.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    time.sleep(1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--readback',action='store_true');p.add_argument('--repair-missing',action='store_true');a=p.parse_args()
    if a.repair_missing:repair_missing(a.directory)
    export(a.directory,readback=a.readback)

if __name__=='__main__':main()
