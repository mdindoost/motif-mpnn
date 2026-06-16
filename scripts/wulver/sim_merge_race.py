#!/usr/bin/env python3
"""
sim_merge_race.py — Empirical validation of the Arachne subgraph-isomorphism
merge-race bug fix, via a faithful Python thread simulation.

WHY THIS EXISTS
---------------
There is no Chapel compiler on this machine. The bug under review lives in
Chapel parallel code (Arachne / HiPerMotif), where subgraph-isomorphism matches
are merged into a SHARED `parSafe=true` list. We cannot compile the real thing,
so we reproduce the *exact concurrency semantics* with real OS threads forced to
interleave, and measure the failure mode and its repair.

MODEL OF THE REAL SYSTEM
------------------------
- A match is a contiguous block of k = numSubgraphVertices host-vertex ids
  (e.g. k=3 wedge: [centerNbr, center, otherNbr]).
- Many parallel tasks each hold a local `newMappings` list of whole k-int blocks
  and merge it into the shared flat list. The flat list is later reshaped to
  (-1, k); each row must be ONE intact match.

- ORIGINAL (buggy):
      for m in newMappings do shared.pushBack(m)        // element by element
  `parSafe=true` => each SINGLE pushBack is atomic (no torn/lost element), but
  the loop is NOT atomic as a whole. Task A's [a0,a1,a2] can have task B's ints
  land between them. Total element count stays exact (every element pushed once),
  but reshape(-1,k) yields scrambled rows, including repeated-id rows.

- FIXED (patch under review):
      appendLock.acquire(); for m in newMappings do shared.pushBack(m); appendLock.release()
  The whole per-task block-set is appended in one critical section -> contiguous.

HOW THE SIMULATION IS FAITHFUL
------------------------------
Buggy model:  `_AtomicList.append_one_atomic(x)` takes a lock for EXACTLY ONE
element (mirrors parSafe single-pushBack atomicity). The per-task python loop
releases that lock between elements and yields the GIL (time.sleep(0)) so other
threads' single appends interleave. This is precisely "single-pushBack atomic,
multi-element loop not atomic".

Fixed model: each task acquires ONE `block_lock` and holds it across its entire
block-set append (mirrors the Chapel mutex / defer-released lock). No element of
another task can land inside.

Detection: thread t's blocks use ids tagged (thread, block, position) so every
id in the whole run is globally UNIQUE. A correct row therefore has:
  (a) k ids, all from the SAME (thread, block) source, and
  (b) positions 0..k-1 in order.
Any deviation (ids from >1 source block, or a repeated/duplicated id within a
row) marks a scrambled row. We also independently verify the total element count
and that the multiset of all ids is conserved (nothing lost or duplicated).

GIL CAVEAT (so we neither over- nor under-claim)
------------------------------------------------
CPython's GIL serializes bytecode, so two python appends never physically race
on memory the way Chapel tasks on distinct cores do. We are NOT relying on a
hardware data race. We are modeling the *logical* interleaving that parSafe
guarantees at the granularity of a single pushBack: each single append is atomic
(true in both Chapel-parSafe and under-GIL+lock), and the scheduler is free to
interleave whole loops. The GIL + explicit time.sleep(0) yields make interleaving
MORE deterministic/reliable here than on a cluster (good for reproducibility),
but the *kind* of interleaving — single appends from different tasks landing
between each other — is exactly what parSafe permits in Chapel. So this faithfully
reproduces the failure mode; it does not, and need not, reproduce a sub-pushBack
data race (parSafe forbids that in Chapel too).
"""

import threading
import time
import sys
from collections import Counter


# ---------------------------------------------------------------------------
# Shared list models
# ---------------------------------------------------------------------------

class AtomicList:
    """Models a Chapel list(parSafe=true): each SINGLE append is mutually
    exclusive (atomic), but nothing larger is."""

    def __init__(self):
        self._data = []
        self._elem_lock = threading.Lock()   # guards one append (parSafe)
        self.block_lock = threading.Lock()    # used ONLY by the fixed merge

    def append_one_atomic(self, x):
        # Exactly one element under the lock -> mirrors parSafe single pushBack.
        with self._elem_lock:
            self._data.append(x)

    @property
    def data(self):
        return self._data


# ---------------------------------------------------------------------------
# The two merge strategies (run by every worker thread)
# ---------------------------------------------------------------------------

def buggy_merge(shared, new_mappings, yield_between=True):
    """ORIGINAL: for m in newMappings do shared.pushBack(m).
    Lock released between elements; yield so other tasks interleave."""
    for block in new_mappings:
        for m in block:
            shared.append_one_atomic(m)
            if yield_between:
                time.sleep(0)   # release GIL -> scheduler may run other threads


def fixed_merge(shared, new_mappings, yield_between=True):
    """FIXED: appendLock.acquire(); for m in newMappings do pushBack(m); release().
    Whole per-task block-set under one lock => contiguous, even with yields."""
    with shared.block_lock:
        for block in new_mappings:
            for m in block:
                shared.append_one_atomic(m)
                if yield_between:
                    time.sleep(0)   # yields still happen, but lock is held


# ---------------------------------------------------------------------------
# Build per-thread work: globally-unique tagged ids
# ---------------------------------------------------------------------------

def make_blocks(thread_id, blocks_per_thread, k):
    """Return a list of blocks; each id is a tuple (thread, block, pos) so it is
    globally unique and a correct row is fully identifiable."""
    blocks = []
    for b in range(blocks_per_thread):
        blocks.append([(thread_id, b, pos) for pos in range(k)])
    return blocks


def run_trial(merge_fn, num_threads, blocks_per_thread, k):
    shared = AtomicList()
    work = {t: make_blocks(t, blocks_per_thread, k) for t in range(num_threads)}

    # Barrier so all threads enter the merge region as close to simultaneously
    # as possible (maximizes interleaving for the buggy case).
    barrier = threading.Barrier(num_threads)

    def worker(t):
        nm = work[t]
        barrier.wait()
        merge_fn(shared, nm)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    return shared.data


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(flat, num_threads, blocks_per_thread, k):
    expected_count = num_threads * blocks_per_thread * k
    count_ok = (len(flat) == expected_count)

    # Multiset conservation: every (thread,block,pos) id should appear exactly once.
    expected_ms = Counter()
    for t in range(num_threads):
        for b in range(blocks_per_thread):
            for pos in range(k):
                expected_ms[(t, b, pos)] += 1
    actual_ms = Counter(flat)
    multiset_ok = (actual_ms == expected_ms)

    # Reshape into rows of k. (If count not divisible by k, count remainder
    # as automatically scrambled — but count is exact in both models.)
    n_full = len(flat) // k
    scrambled = 0
    for r in range(n_full):
        row = flat[r * k:(r + 1) * k]
        sources = {(tid, blk) for (tid, blk, pos) in row}
        positions = [pos for (tid, blk, pos) in row]
        ids_unique = (len(set(row)) == k)
        intact = (
            len(sources) == 1 and          # all from one (thread,block)
            positions == list(range(k)) and  # positions 0..k-1 in order
            ids_unique                      # no repeated id within row
        )
        if not intact:
            scrambled += 1
    # leftover elements that don't fill a row (shouldn't happen here)
    if len(flat) % k != 0:
        scrambled += 1

    return {
        "count": len(flat),
        "expected_count": expected_count,
        "count_ok": count_ok,
        "multiset_ok": multiset_ok,
        "scrambled_rows": scrambled,
        "total_rows": n_full,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    NUM_THREADS = 32
    BLOCKS_PER_THREAD = 8
    K = 3                # wedge: [centerNbr, center, otherNbr]
    TRIALS = 8

    print("=" * 78)
    print("Arachne merge-race simulation  (faithful Chapel-semantics model)")
    print("=" * 78)
    print(f"threads={NUM_THREADS}  blocks/thread={BLOCKS_PER_THREAD}  k={K}  "
          f"trials={TRIALS}")
    print(f"expected total elements per trial = "
          f"{NUM_THREADS*BLOCKS_PER_THREAD*K}  "
          f"(rows = {NUM_THREADS*BLOCKS_PER_THREAD})")
    print()

    for label, fn in (("BUGGY", buggy_merge), ("FIXED", fixed_merge)):
        print(f"--- {label} merge "
              f"({'element-by-element, lock released between elements' if label=='BUGGY' else 'whole block-set under one lock'}) ---")
        header = (f"{'trial':>5} | {'count_ok':>8} | {'count':>6}/{'exp':<6} | "
                  f"{'multiset_ok':>11} | {'scrambled_rows':>14} | {'/total':>7}")
        print(header)
        print("-" * len(header))
        scrambled_seq = []
        for trial in range(TRIALS):
            flat = run_trial(fn, NUM_THREADS, BLOCKS_PER_THREAD, K)
            res = analyze(flat, NUM_THREADS, BLOCKS_PER_THREAD, K)
            scrambled_seq.append(res["scrambled_rows"])
            print(f"{trial:>5} | {str(res['count_ok']):>8} | "
                  f"{res['count']:>6}/{res['expected_count']:<6} | "
                  f"{str(res['multiset_ok']):>11} | "
                  f"{res['scrambled_rows']:>14} | "
                  f"{res['total_rows']:>7}")
        print(f"  scrambled_rows across trials: {scrambled_seq}  "
              f"(min={min(scrambled_seq)}, max={max(scrambled_seq)}, "
              f"varying={len(set(scrambled_seq))>1})")
        print()

    # ---- An illustrative scrambled row from one buggy trial -----------------
    print("--- illustrative buggy output (one trial, first few scrambled rows) ---")
    flat = run_trial(buggy_merge, NUM_THREADS, BLOCKS_PER_THREAD, K)
    n_full = len(flat) // K
    shown = 0
    for r in range(n_full):
        row = flat[r * K:(r + 1) * K]
        sources = {(tid, blk) for (tid, blk, pos) in row}
        if len(sources) != 1 or [p for (_, _, p) in row] != list(range(K)):
            # render as compact "t.b.pos" tags
            rendered = [f"{t}.{b}.{p}" for (t, b, p) in row]
            print(f"  row {r}: {rendered}   sources={sorted(sources)}")
            shown += 1
            if shown >= 5:
                break
    if shown == 0:
        print("  (no scrambled rows in this particular trial)")
    print()


if __name__ == "__main__":
    sys.exit(main())
