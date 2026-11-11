"""Append-only local E2E events/spans/receipts and low-cardinality Prometheus export."""
from __future__ import annotations
from collections import Counter, defaultdict
from contextlib import contextmanager
import contextvars
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import threading
import time
import uuid

CURRENT = contextvars.ContextVar('finder_span', default=None)
BUCKETS = (.001,.005,.01,.025,.05,.1,.25,.5,1,2.5,5,10,30,60,180,float('inf'))
SENSITIVE = re.compile(r'authorization|api.?key|password|secret|access.?token|reasoning(?:_content)?|error_model_output', re.I)


def clean(value):
    if isinstance(value,dict):
        return {k:'[redacted]' if SENSITIVE.search(k) else clean(v) for k,v in value.items()}
    if isinstance(value,list):return [clean(v) for v in value]
    if isinstance(value,str):
        value=re.sub(r'<\|channel\|>analysis<\|message\|>.*?(?=<\|channel\|>final<\|message\|>|$)','[private reasoning omitted]',value,flags=re.S|re.I)
        return re.sub(r'<(?:mm:)?think>.*?(?:</(?:mm:)?think>|$)','[private reasoning omitted]',value,flags=re.S)
    return value


def encoded(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()


class Telemetry:
    def __init__(self, directory, run_id):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.run_id=run_id;self.lock=threading.RLock()
        (self.directory/'receipts').mkdir(exist_ok=True)
        self.counts=Counter();self.sums=defaultdict(float);self.hist=defaultdict(Counter)
        # No implicit resume that would silently reset metrics. Rehydrate from canonical spans.
        if (self.directory/'spans.jsonl').exists():
            for line in (self.directory/'spans.jsonl').read_text().splitlines():
                self._observe(json.loads(line))

    def append(self, name, record):
        payload=encoded(clean(record))+b'\n'
        with self.lock:
            with (self.directory/name).open('ab') as stream:
                stream.write(payload);stream.flush();os.fsync(stream.fileno())

    def append_batch(self, name, records):
        """Persist an explicitly buffered batch outside a timed load window."""
        payload=b''.join(encoded(clean(record))+b'\n' for record in records)
        with self.lock:
            with (self.directory/name).open('ab') as stream:
                stream.write(payload);stream.flush();os.fsync(stream.fileno())

    def receipt(self, value):
        payload=encoded(clean(value));sha=hashlib.sha256(payload).hexdigest()
        path=self.directory/'receipts'/f'{sha}.json'
        with self.lock:
            if path.exists():
                if path.read_bytes()!=payload:raise ValueError('receipt hash collision')
            else:
                with path.open('xb') as stream:
                    stream.write(payload);stream.flush();os.fsync(stream.fileno())
                path.chmod(0o600)
        return sha

    def event(self, name, fields=None):
        context=CURRENT.get() or {}
        self.append('events.jsonl',{'schema_version':1,'run_id':self.run_id,
                    'utc':datetime.now(timezone.utc).isoformat(),'monotonic_ns':time.perf_counter_ns(),
                    **context,'event':name,'fields':fields or {}})

    @contextmanager
    def span(self, component, episode_id, endpoint_id=None, attempt=0):
        if not re.fullmatch(r'[a-z][a-z0-9_]{0,63}',component):raise ValueError('invalid component')
        parent=CURRENT.get()
        if parent and parent['episode_id']!=episode_id:raise ValueError('nested episode mismatch')
        context={'episode_id':episode_id,'trace_id':parent['trace_id'] if parent else uuid.uuid4().hex,
                 'span_id':uuid.uuid4().hex}
        token=CURRENT.set(context);start=time.perf_counter_ns()
        record={'schema_version':1,'run_id':self.run_id,**context,'parent_span_id':parent['span_id'] if parent else None,
                'component':component,'attempt':attempt,'status':'ok','duration_ms':0.,'endpoint_id':endpoint_id,
                'receipt_sha256':None,'measurements':dict.fromkeys(['http_status','structured_code','input_tokens','output_tokens','ttft_ms','db_hits','db_page_faults'])}
        record['measurements']['missing_reasons']={k:'not_observed' for k in record['measurements']}
        self.event('stage_started',{'component':component,'attempt':attempt})
        try:
            yield record
        except BaseException as error:
            record['status']='timeout' if isinstance(error,TimeoutError) else 'error'
            self.event('stage_exception',{'component':component,'exception_type':type(error).__name__})
            raise
        finally:
            record['duration_ms']=(time.perf_counter_ns()-start)/1e6
            # Metrics with values should not retain a stale missingness reason.
            for key,value in record['measurements'].items():
                if key!='missing_reasons' and value is not None:record['measurements']['missing_reasons'].pop(key,None)
            try:
                self.append('spans.jsonl',record)
                with self.lock:self._observe(record)
                self.event('stage_finished',{'component':component,'status':record['status']})
            finally:CURRENT.reset(token)

    def _observe(self, record):
        key=(record['component'],record['status'])
        seconds=record['duration_ms']/1000
        if not math.isfinite(seconds) or seconds<0:raise ValueError('invalid duration')
        self.counts[key]+=1;self.sums[key]+=seconds
        for bound in BUCKETS:
            if seconds<=bound:self.hist[key][bound]+=1

    def prometheus(self):
        lines=['# TYPE finder_stage_duration_seconds histogram','# TYPE finder_stage_total counter']
        with self.lock:
            for (component,status),count in sorted(self.counts.items()):
                # Only bounded component/status; no per-episode, text, entity or query labels.
                labels=f'component="{component}",status="{status}"'
                lines.extend([f'finder_stage_total{{{labels}}} {count}',
                              f'finder_stage_duration_seconds_count{{{labels}}} {count}',
                              f'finder_stage_duration_seconds_sum{{{labels}}} {self.sums[(component,status)]}'])
                for bound in BUCKETS:
                    label='+Inf' if math.isinf(bound) else str(bound)
                    lines.append(f'finder_stage_duration_seconds_bucket{{{labels},le="{label}"}} {self.hist[(component,status)][bound]}')
        return '\n'.join(lines)+'\n'

    def flush_metrics(self):
        with self.lock:
            temporary=self.directory/'metrics.prom.tmp'
            temporary.write_text(self.prometheus());temporary.replace(self.directory/'metrics.prom')

    def external_span(self, component, episode_id, endpoint_id, started_unix_ns, duration_ns, receipt_sha256, structured_code=None, *, timestamp_origin='rust_systemtime_at_job_start_microsecond_projection', duration_origin='rust_process_monotonic'):
        """Import an actual externally timed child; never substitute parent/client timings."""
        parent=CURRENT.get()
        if not parent or parent['episode_id']!=episode_id:raise ValueError('external span requires matching active parent')
        if not re.fullmatch(r'[a-z][a-z0-9_]{0,63}',component):raise ValueError('invalid component')
        if type(started_unix_ns) is not int or started_unix_ns<=0 or type(duration_ns) is not int or duration_ns<0:raise ValueError('invalid external clocks')
        context={'episode_id':episode_id,'trace_id':parent['trace_id'],'span_id':uuid.uuid4().hex}
        measurements=dict.fromkeys(['http_status','structured_code','input_tokens','output_tokens','ttft_ms','db_hits','db_page_faults'])
        measurements['structured_code']=structured_code
        measurements['missing_reasons']={k:('not_profiled' if k.startswith('db_') else 'not_a_provider_measurement') for k,v in measurements.items() if v is None}
        record={'schema_version':1,'run_id':self.run_id,**context,'parent_span_id':parent['span_id'],'component':component,'attempt':0,'status':'error' if structured_code else 'ok','duration_ms':duration_ns/1e6,'endpoint_id':endpoint_id,'receipt_sha256':receipt_sha256,'measurements':measurements}
        started=datetime.fromtimestamp(started_unix_ns/1e9,timezone.utc).isoformat()
        self.append('events.jsonl',{'schema_version':1,'run_id':self.run_id,**context,'utc':started,'monotonic_ns':None,'event':'stage_started','fields':{'component':component,'timestamp_origin':timestamp_origin,'duration_origin':duration_origin}})
        self.append('spans.jsonl',record)
        with self.lock:self._observe(record)
        return record

    def host_sample(self):
        own=resource.getrusage(resource.RUSAGE_SELF)
        sample={'run_id':self.run_id,'utc':datetime.now(timezone.utc).isoformat(),
                'process_user_cpu_s':own.ru_utime,'process_system_cpu_s':own.ru_stime,
                'process_max_rss_kib':own.ru_maxrss,'process_minor_faults':own.ru_minflt,
                'process_major_faults':own.ru_majflt,'process_read_io_bytes':None,'process_write_io_bytes':None,
                'server_page_faults':None,'server_page_faults_reason':'requires_server_collector'}
        path=Path('/proc/self/io')
        if path.exists():
            fields=dict(line.split(': ',1) for line in path.read_text().splitlines())
            sample['process_read_io_bytes']=int(fields['read_bytes']);sample['process_write_io_bytes']=int(fields['write_bytes'])
        self.append('resources.jsonl',sample)
        return sample


def audit(directory):
    directory=Path(directory)
    redactions=json.loads((directory/'receipt-redactions.json').read_text()) if (directory/'receipt-redactions.json').exists() else {}
    spans=[json.loads(line) for line in (directory/'spans.jsonl').read_text().splitlines() if line.strip()]
    ids={r['span_id']:r for r in spans};errors=[];receipt_validity={}
    if len(ids)!=len(spans):errors.append('duplicate_span_id')
    for row in spans:
        parent=ids.get(row['parent_span_id'])
        if row['parent_span_id'] and not parent:errors.append('orphan_parent')
        if parent and (parent['trace_id']!=row['trace_id'] or parent['episode_id']!=row['episode_id']):errors.append('parent_identity_mismatch')
        sha=row['receipt_sha256']
        if sha:
            resolved=redactions.get(sha,{}).get('replacement_sha256',sha)
            if resolved not in receipt_validity:
                p=directory/'receipts'/f'{resolved}.json'
                receipt_validity[resolved]=p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==resolved
            if not receipt_validity[resolved]:errors.append('missing_or_changed_receipt')
    return {'span_count':len(spans),'errors':errors,'valid':not errors,
            'scope':'trace_integrity_only_not_full_stage_or_semantic_readiness'}
