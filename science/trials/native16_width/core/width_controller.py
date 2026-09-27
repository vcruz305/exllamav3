"""One-shot, dependency-injected controller; no automatic candidate launches.

Local tests exercise the complete outer identity -> inner pre-stop -> retained
launch path. An owner-supplied adapter must implement the protocol in README.
No default live adapter/CLI is supplied: this local package is NOT DEPLOYED.
"""
import recover

def identity(expected,io):
    return recover.identity(expected,io)

def launch(expected,io,**timeouts):
    before=identity(expected,io)
    io.before_stop()
    memory=recover.stop_and_release(expected,io,**timeouts)
    # Single invocation only; failure never retries a large-model load.
    receipt=io.launch_retained()
    ready=io.wait_retained(receipt)
    if ready.get('healthy') is not True or type(ready.get('requests')) is not int or ready['requests']!=0:
        recover.refuse('retained_not_ready',ready)
    if not ready.get('run') or ready['run']==expected['run']:
        recover.refuse('retained_identity_unproved',ready)
    return {'before':before,'memory':memory,'retained':ready}

def run_once(width_gate,expected,io,save_status,**timeouts):
    status={'width':'not_run','rollback':'not_run','width_error':None,'rollback_error':None}
    try:
        width_gate()
        status['width']='passed'
    except BaseException as exc:
        status['width']='failed';status['width_error']=repr(exc)
    finally:
        # Preserve the width outcome independently of recovery success or failure.
        try:
            status['recovery']=launch(expected,io,**timeouts)
            status['rollback']='restored'
        except BaseException as exc:
            status['rollback']='failed';status['rollback_error']=repr(exc)
        save_status(status)
    return 0 if status['width']=='passed' and status['rollback']=='restored' else 1
