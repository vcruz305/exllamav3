"""Fail-closed recovery core. All external boundaries explicit; no import-time I/O.

An exclusive owner must supply a trusted PRE-FAILURE PID/starttime/argv receipt.
The adapter is intentionally not auto-discovered: no receipt means no STOP/load.
This module does not send signals, invoke SSH, or select a candidate launcher.
"""
import json
import socket
import urllib.error

class Handoff(RuntimeError):
    pass

def refuse(code, detail=''):
    raise Handoff(code+': '+str(detail)+'; keep exclusive ownership, inspect exact PID/starttime/argv and request state; do not broad-kill or retry candidate')

def health(io):
    try:
        try:
            response=io.open_health('http://127.0.0.1:8096/health',timeout=5)
            status=response.status
        except urllib.error.HTTPError as exc:
            response=exc;status=exc.code
        with response:
            raw=response.read()
    except (socket.timeout,TimeoutError) as exc:
        refuse('timeout',exc)
    except urllib.error.URLError as exc:
        if isinstance(exc.reason,(socket.timeout,TimeoutError)):refuse('timeout',exc)
        if isinstance(exc.reason,ConnectionRefusedError):refuse('connection_refused',exc)
        refuse('transport_error',exc)
    except ConnectionRefusedError as exc:
        refuse('connection_refused',exc)
    try:
        body=json.loads(raw)
    except (ValueError,UnicodeError) as exc:
        refuse('malformed_body',exc)
    if not isinstance(body,dict):refuse('malformed_body','expected object')
    if status not in (200,503):refuse('unexpected_http_status',status)
    if type(body.get('requests')) is not int or body['requests']<0:refuse('unknown_requests',body)
    if body['requests']!=0:refuse('active_requests',body)
    if type(body.get('healthy')) is not bool:refuse('unknown_health',body)
    return {'status':status,'body':body}

def identity(expected,io):
    try:
        rows=expected['processes'];pids=expected['pids'];run=expected['run']
        if len(rows)!=2 or [r['pid'] for r in rows]!=[pids['guard_pid'],pids['server_pid']]:refuse('invalid_owner_receipt')
        if rows[0]['pid']==rows[1]['pid']:refuse('invalid_owner_receipt','duplicate PID')
        for row in rows:
            if type(row['pid']) is not int or row['pid']<=0 or not str(row['starttime']).isdigit() or not row['args']:refuse('invalid_owner_receipt')
        if rows[1]['args']!=pids['command']:refuse('invalid_owner_receipt','server argv')
        if io.active()!=run or io.pids(run)!=pids:refuse('identity_changed','active run or pid record')
        for row in rows:
            if io.process(row['pid'])!=row:refuse('identity_changed',row['pid'])
    except (KeyError,TypeError,ValueError,OSError) as exc:
        refuse('identity_unproved',exc)
    return {'run':run,'processes':rows,'health':health(io)}

def pre_stop(expected,io):
    # Real second outer/inner verification: changes or new requests abort before STOP.
    identity(expected,io)
    message='Owned native-width diagnostic recovery; retained launcher only.\n'
    io.write_stop(expected['run'],message)
    if io.read_stop(expected['run'])!=message:refuse('STOP_readback')

def stop_and_release(expected,io,stop_seconds=120,headroom_seconds=120):
    pre_stop(expected,io)
    deadline=io.monotonic()+stop_seconds
    while True:
        live=[]
        for row in expected['processes']:
            current=io.process(row['pid'])
            if current is not None and current!=row:refuse('identity_changed_after_STOP',row['pid'])
            live.append(current is not None)
        if not any(live) and io.result(expected['run']) is not None:break
        if io.monotonic()>=deadline:refuse('stop_timeout','no overlapping load allowed')
        io.sleep(.5)
    deadline=io.monotonic()+headroom_seconds
    while True:
        m=io.memory()
        required=('available_gib','cgroup_headroom_gib','free_gib','host_oom_kill','cgroup_oom','cgroup_oom_kill')
        if any(k not in m for k in required):refuse('memory_unproved',m)
        if any(m[k]!=0 for k in ('host_oom_kill','cgroup_oom','cgroup_oom_kill')):refuse('OOM_counter',m)
        # Existing load-headroom policy, not a change to independent 8GiB runtime floors.
        if m['available_gib']>=102 and m['cgroup_headroom_gib']>=102 and m['free_gib']>=2:break
        if io.monotonic()>=deadline:refuse('headroom_timeout',m)
        io.sleep(1)
    if io.active()!=expected['run']:refuse('identity_changed','active run changed after STOP')
    if any(io.process(r['pid']) is not None for r in expected['processes']):refuse('identity_changed','owned PID reappeared')
    return m
