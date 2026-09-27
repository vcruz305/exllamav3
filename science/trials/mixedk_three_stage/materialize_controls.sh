#!/bin/sh
# Materialize the three source trees this trial is verified against.
#
# None of them are vendored in this branch: each is a complete copy of the library
# sources (~21 MB apiece, ~2,100 files), so they are rebuilt here from git history and
# stay out of the commit.
#
#   original/            the unpatched sources, exactly BASE_SHA
#   tree/                the candidate as committed on this branch (git archive HEAD)
#   five-stage-control/  the historical five-stage candidate:
#                        BASE_SHA + five-stage-control.delta.patch (4 paths)
#
# Usage (from anywhere):
#     sh science/trials/mixedk_three_stage/materialize_controls.sh
#
# Then run the CPU suite; see README.md for the full invocation order.
#
# Notes
#   * BASE_SHA (ca4a880) and origin/master (74b6f5a) are identical under exllamav3/, so
#     basing this branch on master does not change what the trial compares.
#   * five-stage-control could not be produced by `git archive` alone: that experiment
#     was never committed to any git history. Only its 4-path delta against BASE_SHA is
#     shipped (a 17 KB patch), which this script applies to reproduce it exactly.

set -eu

BASE_SHA=ca4a880e8918e1985fd25e06c6aff561666d3f14
DELTA_PATCH=five-stage-control.delta.patch

# sha256 of each materialized tree, as defined by the digest below:
# sha256 over sorted "path<TAB>blob-sha1\n" lines, __pycache__ excluded.
ORIGINAL_TREE_SHA256=11f05c4760d0563abca13665cf9dbf7a38045aa7f6244c5c58007557505fe7c4
TREE_SHA256=6be6822a24cecabeb0631fa9db044ece595f99da00a68f367d2c88b533196b3f
FIVE_STAGE_TREE_SHA256=048c8d56715fdd8ba29f7229b61bfdc700caf2bb6a5035335eb7580b8d43f0bf

# This host runs git and python as native Windows binaries (no path translation), while
# tar is an MSYS binary. Keep both spellings of the directory.
HERE=$(cd "$(dirname "$0")" && pwd)
HERE_NATIVE=$(cd "$(dirname "$0")" && pwd -W 2>/dev/null || printf '%s' "$HERE")
REPO=$(git -C "$HERE_NATIVE" rev-parse --show-toplevel)

if ! git -C "$REPO" cat-file -e "$BASE_SHA^{commit}" 2>/dev/null; then
    echo "ERROR: commit $BASE_SHA is not present in $REPO" >&2
    echo "       fetch it first (it is an ancestor of origin/master)." >&2
    exit 1
fi

echo "base sha     : $BASE_SHA"
echo "repo         : $REPO"
echo "trees        : original/            = BASE_SHA"
echo "               tree/                = HEAD ($(git -C "$REPO" rev-parse --short=12 HEAD))"
echo "               five-stage-control/  = BASE_SHA + $DELTA_PATCH"

# --- original/ --------------------------------------------------------------
rm -rf "$HERE/original"
mkdir -p "$HERE/original"
git -C "$REPO" archive --format=tar "$BASE_SHA" | tar -x -C "$HERE/original"
echo "materialized : original/            ($(find "$HERE/original" -type f | wc -l | tr -d ' ') files)"

# --- tree/ (the candidate as committed) -------------------------------------
# BASE_SHA gives the whole repository; the candidate patch only ever touches
# exllamav3/, so overlaying this branch's exllamav3/ onto BASE_SHA reproduces the
# prepared candidate tree exactly.
rm -rf "$HERE/tree"
mkdir -p "$HERE/tree"
git -C "$REPO" archive --format=tar "$BASE_SHA" | tar -x -C "$HERE/tree"
rm -rf "$HERE/tree/exllamav3"
git -C "$REPO" archive --format=tar HEAD exllamav3 | tar -x -C "$HERE/tree"
echo "materialized : tree/                ($(find "$HERE/tree" -type f | wc -l | tr -d ' ') files)"

# --- five-stage-control/ ----------------------------------------------------
rm -rf "$HERE/five-stage-control"
mkdir -p "$HERE/five-stage-control"
git -C "$REPO" archive --format=tar "$BASE_SHA" | tar -x -C "$HERE/five-stage-control"
(
    cd "$HERE/five-stage-control"
    git apply -p1 "../$DELTA_PATCH"
)
echo "materialized : five-stage-control/  ($(find "$HERE/five-stage-control" -type f | wc -l | tr -d ' ') files)"

# --- self-check -------------------------------------------------------------
# Recomputes the tree digests and compares them to the values recorded above.
if command -v python3 >/dev/null 2>&1; then PY=python3
elif command -v python >/dev/null 2>&1; then PY=python
else PY=
fi

if [ -n "$PY" ]; then
    "$PY" - "$HERE_NATIVE" "$ORIGINAL_TREE_SHA256" "$TREE_SHA256" "$FIVE_STAGE_TREE_SHA256" <<'EOF'
import hashlib, sys
from pathlib import Path

here, want_orig, want_tree, want_five = (Path(sys.argv[1]), sys.argv[2],
                                         sys.argv[3], sys.argv[4])

def digest(root):
    h = hashlib.sha256()
    entries = []
    for p in root.rglob("*"):
        if p.is_file() and "__pycache__" not in p.parts:
            data = p.read_bytes()
            entries.append((p.relative_to(root).as_posix(),
                            hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()))
    for path, blob in sorted(entries):
        h.update(path.encode()); h.update(b"\t")
        h.update(blob.encode()); h.update(b"\n")
    return h.hexdigest(), len(entries)

failed = False
for name, sub, want in (("original", "", want_orig),
                        ("tree", "", want_tree),
                        ("five-stage-control", "", want_five)):
    got, n = digest(here / name / sub)
    ok = got == want
    failed |= not ok
    print(f"  {'OK  ' if ok else 'FAIL'} {name + '/':<21} tree digest {got} ({n} files)")
    if not ok:
        print(f'       expected {want}')
print("SELF-CHECK FAILED" if failed else "SELF-CHECK PASSED")
sys.exit(1 if failed else 0)
EOF
else
    echo "python not found; verify by hand against:"
    echo "  original/           $ORIGINAL_TREE_SHA256"
    echo "  tree/exllamav3/     $TREE_SHA256"
    echo "  five-stage-control/ $FIVE_STAGE_TREE_SHA256"
fi
