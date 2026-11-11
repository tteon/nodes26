"""Bounded MARA HTTP attempts with immutable receipts and unresolved-outcome guards."""
from __future__ import annotations
import json
import fcntl
import os
from pathlib import Path
import random
import time
import urllib.error
import urllib.request
from telemetry import Telemetry,clean,encoded

RETRY={408,429,500,503}


def credential(path):
    values={}
    if path:
        for line in Path(path).read_text().splitlines():
            line=line.strip().removeprefix('export ')
            if '=' in line and not line.startswith('#'):
                key,value=line.split('=',1);values[key.strip()]=value.strip().strip('\"\'')
    key=os.environ.get('MARA_API_KEY') or values.get('MARA_API_KEY')
    if not key:raise RuntimeError('MARA_API_KEY unavailable')
    return key


class Mara:
    def __init__(self,telemetry,credential_file=None,base='https://api.cloud.mara.com/v1'):
        self.telemetry=telemetry;self.key=credential(credential_file);self.base=base.rstrip('/')
        if not self.base.startswith('https://'):raise ValueError('HTTPS required')
        self.state=telemetry.directory/'http-state';self.state.mkdir(exist_ok=True)

    def request(self,episode,request_id,body=None,path='/chat/completions',*,deadline=None,max_attempts=4):
        # Cross-run process lock keeps the declared initial MARA concurrency at one.
        with (self.telemetry.directory.parent/'.mara-global.lock').open('a+') as lock:
            with self.telemetry.span('mara_admission',episode):
                fcntl.flock(lock,fcntl.LOCK_EX)
            try:
                if any(self.telemetry.directory.parent.glob('*/http-state/*.pending')):
                    raise RuntimeError('Outstanding or unresolved HTTP outcome; inspect pending receipts before further admission')
                return self._request(episode,request_id,body,path,stage_budget=90 if deadline is None else min(90,max(0,deadline-time.monotonic())),max_attempts=max_attempts)
            finally:fcntl.flock(lock,fcntl.LOCK_UN)

    def _request(self,episode,request_id,body=None,path='/chat/completions',*,stage_budget=90,max_attempts=4):
        if not 1<=max_attempts<=4:raise ValueError('max_attempts must be 1..4')
        import hashlib
        if not request_id.replace('-','').replace('_','').isalnum():raise ValueError('invalid request ID')
        payload=encoded(body) if body is not None else None
        identity=hashlib.sha256(encoded({'path':path,'body':body})).hexdigest()
        saved=self.state/f'{request_id}.json';pending=self.state/f'{request_id}.pending'
        if saved.exists():
            result=json.loads(saved.read_text())
            if result['request_sha256']!=identity:raise ValueError('request ID reused for changed body')
            self.telemetry.event('http_cached_receipt',{'request_id':request_id})
            return result
        if pending.exists():raise RuntimeError('Unresolved HTTP outcome; inspect pending receipt before resubmission')
        started=time.monotonic();result=None
        for attempt in range(max_attempts):
            if time.monotonic()-started>=stage_budget:break
            with pending.open('x') as stream:json.dump({'request_sha256':identity,'attempt':attempt,'path':path},stream)
            status=None;response=None;network_error=None
            with self.telemetry.span('mara',episode,endpoint_id='mara',attempt=attempt) as span:
                request=urllib.request.Request(self.base+path,data=payload,headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'})
                try:
                    with urllib.request.urlopen(request,timeout=min(30,stage_budget-(time.monotonic()-started))) as remote:
                        status=remote.status;raw=remote.read()
                except urllib.error.HTTPError as error:
                    status=error.code;raw=error.read()
                except (urllib.error.URLError,TimeoutError) as error:
                    network_error=type(error).__name__;raw=b''
                try:response=json.loads(raw) if raw else None
                except ValueError:response={'unparsed_body':raw.decode('utf-8',errors='replace')[:8000]}
                if response is not None:response=clean(response)
                error=(response or {}).get('error') or {}
                code=error.get('code') if isinstance(error,dict) else None
                usage=(response or {}).get('usage') or {}
                receipt={'request_id':request_id,'request_sha256':identity,'path':path,'request':body,
                         'attempt':attempt,'http_status':status,'structured_code':code,
                         'network_error':network_error,'response':response,'usage':usage,'actual_billed_usd':None}
                sha=self.telemetry.receipt(receipt)
                function=(((body or {}).get('tool_choice') or {}).get('function') or {}).get('name') if isinstance((body or {}).get('tool_choice'),dict) else None
                role='index' if function in ['submit_facts','submit_observations'] else 'answer' if function in ['submit_answer','submit_results'] else 'tool_request' if function=='fetch_attached_evidence' else 'catalog' if path=='/models' else 'health'
                span['model_id']=(body or {}).get('model')
                span['request_role']=role
                span.update(receipt_sha256=sha,status='ok' if status==200 else 'error')
                span['measurements'].update(http_status=status,structured_code=code,
                     input_tokens=usage.get('prompt_tokens'),output_tokens=usage.get('completion_tokens'))
                result={'request_sha256':identity,'receipt_sha256':sha,'http_status':status,'response':response,'attempt':attempt}
            # Persist received result before clearing pending. Unknown transport outcome is not retried.
            if network_error:
                result['unresolved']=True
                return result
            pending.unlink()
            if status==200 or status not in RETRY or attempt==max_attempts-1:break
            wait=min(2**attempt+random.random(), max(0.,stage_budget-(time.monotonic()-started)))
            with self.telemetry.span('backoff',episode):time.sleep(wait)
        if result is None:raise TimeoutError('HTTP stage budget exhausted before request')
        with saved.open('x') as stream:json.dump(result,stream)
        return result
