"""TEST WITNESS: inert substitutions for the width-trial CLI.

Loaded only through `EXL3_WIDTH_TRIAL_IO_HOOK=witness.inert_trial_io`, which the
CLI refuses unless the trial document says `inert_fixture: true`. It exists so
that the REAL CLI (argument parsing, sealed-pin verification, document loading,
stage dispatch, receipt writing) can be exercised end-to-end with real local
Linux processes and an ephemeral loopback HTTP server.

It changes exactly two things, both of which are device facts this host cannot
produce locally:
  * `memory()`      -> a fixed headroom sample (the real policy needs ~110 GiB)
  * `guard_parent_pid()` -> the calling process, because the inert launcher is a
    child of this process instead of init; PR_SET_CHILD_SUBREAPER makes the
    reparented inert guard a child of this process, matching production lineage.
Everything else - /proc identity, pidfd absence, flock ownership, listener
inode, HTTP readiness, STOP/result.json, run-based removal - is the sealed
adapter's unchanged code path.
"""
import ctypes
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import linux_adapter as a  # noqa: E402 - path is set above
import trial_io as ti  # noqa: E402

# DrvFS exposes every file as 0777, so the sealed mode predicate cannot be
# satisfied by any fixture tree. The witness therefore rebinds the predicate
# INSIDE THE WITNESS PROCESS ONLY, exactly as the adapter's local fixtures did
# with mock.patch. The sealed source is not touched (see the preservation
# control in test_trial_linux), the CLI refuses this module for any production
# trial document, and every receipt from a hooked run records `io_hook`.
_SEALED_CHECK_PERMISSIONS = a.check_permissions


def _drvfs_inert_bypass(st):
    return None


a.check_permissions = _drvfs_inert_bypass

MEMORY = dict(available_gib=110, cgroup_headroom_gib=110, free_gib=3,
              host_oom_kill=0, cgroup_oom=0, cgroup_oom_kill=0)
PR_SET_CHILD_SUBREAPER = 36


class InertWidthTrialIO(ti.WidthTrialIO):
    """Restore-stage boundary with the two documented inert substitutions."""

    def memory(self):
        return dict(MEMORY)

    def guard_parent_pid(self):
        return os.getpid()


class InertReleaseIO(ti.ReleaseIO):
    """Release-stage boundary with the same two substitutions."""

    def memory(self):
        return dict(MEMORY)

    def guard_parent_pid(self):
        return os.getpid()


def build(stage):
    if ctypes.CDLL(None).prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        raise a.Refusal('inert witness could not install the child subreaper')
    return InertReleaseIO if stage == 'release' else InertWidthTrialIO
