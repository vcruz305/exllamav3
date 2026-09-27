#!/usr/bin/env python3
"""Run with /usr/bin/python3 -I -B. Adapter bytes validated before import/exec."""
import hashlib
import json
from pathlib import Path
import sys
import types


def main():
    try:
        root = Path(__file__).resolve().parent
        pins = json.loads((root / 'runtime-pins.json').read_text())
        raw = (root / 'linux_adapter.py').read_bytes()
        if hashlib.sha256(raw).hexdigest() != pins['linux_adapter.py']:
            raise RuntimeError('adapter source hash mismatch before load')
        module = types.ModuleType('verified_linux_adapter')
        module.__file__ = str(root / 'linux_adapter.py')
        exec(compile(raw, module.__file__, 'exec'), module.__dict__)
        return module.cli()
    except Exception as error:
        print(json.dumps({'width': 'not_run', 'width_error': None, 'rollback': 'failed',
                          'rollback_error': repr(error), 'readback_error': None}))
        return 1

if __name__ == '__main__':
    sys.exit(main())
