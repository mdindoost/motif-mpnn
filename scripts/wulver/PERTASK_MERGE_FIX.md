# Per-task-local merge — lock-free fix for strong scaling (DRAFT for Bartosz to integrate/compile)

## Why
The race fix added a single global mutex (`appendLock` / `appendBlock`) around every block append.
That keeps each match's block contiguous (correct), but it **serializes the merge** — at high
thread counts every task funnels through one lock → contention → scaling degrades past ~8 cores
(measured: 97% efficiency at 8 cores, then it gets *worse*, 128 slower than 8). This replaces the
global lock with **per-task-local accumulation + a post-`forall` concatenation**, which is both
race-free *and* lock-free → should restore scaling toward 128 cores. (This is the approach the
code reviewers recommended; the lock was the deliberately-minimal first fix.)

> Apply to the **outermost** loops first — `edgeCentricStateInjection` / `vertexCentricStateInjection`
> — that's where almost all the contention is (mG1 ~ millions of iterations through one lock). The
> recursive savers' inner `forall`s are small (few candidate pairs) and lower priority; convert them
> too only if profiling still shows merge cost.

## Driver fix — `edgeCentricStateInjection` (and the same shape for `vertexCentricStateInjection`)

Replace the shared `solutions` list + `appendBlock(...)` inside the `forall` with one result list
per edge (each task writes ONLY its own slot → no sharing during the parallel region → no race AND
no lock), then concatenate sequentially in edge order afterward.

```chapel
proc edgeCentricStateInjection(g1: SegGraph, g2: SegGraph) throws {
  var counts: chpl__processorAtomicType(int) = 0;
  // one private result list PER edge: each task touches only perEdge[edgeIndex], so there is no
  // concurrent access -> parSafe=false, no lock, no race.
  var perEdge: [0..<mG1] list(int, parSafe=false);

  forall edgeIndex in 0..<mG1 with (ref perEdge, ref counts) {
    if limitTime || limitSize then if stopper.read() then continue;
    if vertexFlagger[srcNodesG1[edgeIndex]] && srcNodesG1[edgeIndex] != dstNodesG1[edgeIndex] {
      var initialState = new State(g1.n_vertices, g2.n_vertices);
      var edgeChecked = if findingIsos
        then addToTinToutMVE_ISO(srcNodesG1[edgeIndex], dstNodesG1[edgeIndex], initialState)
        else addToTinToutMVE_MONO(srcNodesG1[edgeIndex], dstNodesG1[edgeIndex], initialState);
      if edgeChecked {
        if countOnly && (limitSize || limitTime || printProgressCheck) {
          counts.add(recursiveMatchCounterVerbose(initialState, 2));
        } else if countOnly {
          counts.add(recursiveMatchCounterFast(initialState, 2));
        } else if !countOnly && (limitSize || limitTime) {
          perEdge[edgeIndex] = recursiveMatchSaverVerbose(initialState, 2);  // whole block, own slot
        } else {
          perEdge[edgeIndex] = recursiveMatchSaverFast(initialState, 2);     // whole block, own slot
        }
      }
    }
  }

  // sequential concat in edge order: O(total output), cheap vs the search; blocks stay contiguous.
  var solutions: list(int);
  if countOnly then solutions.pushBack(counts.read());
  else for e in 0..<mG1 do for m in perEdge[e] do solutions.pushBack(m);

  var subIsoArrToReturn: [0..#solutions.size](int);
  for i in 0..#solutions.size do subIsoArrToReturn[i] = solutions(i);
  return subIsoArrToReturn;
}
```

`vertexCentricStateInjection` is identical except the parallel loop is `forall u in validatedVertices`
with the inner `for v in outNeighbors` — give each `u` its own slot: `var perVertex: [validatedVertices.domain] list(int);` and append the per-(u,v) blocks into `perVertex[u]` inside the (serial) `for v` loop, then concatenate `perVertex` after the `forall`.

## Optional second step — recursive savers (lower priority)
`recursiveMatchSaverFast/Verbose` still call `appendBlock` inside their own `forall (n1,n2) in
candidatePairs`. To make them lock-free too, collect per-candidate then concatenate:

```chapel
proc recursiveMatchSaverFast(state: owned State, depth: int): list(int) throws {
  var allmappings: list(int, parSafe=false);
  if depth == g2.n_vertices { allmappings.pushBack(state.core); return allmappings; }
  var candidatePairs = getCandidatePairsOpti(state);
  // NEEDS CHECK: candidatePairs must be indexable to give each iteration its own slot. If it's a
  // `list`, materialize to an array first (var cps = candidatePairs.toArray();) so perCand can be
  // sized to its domain.
  var cps = candidatePairs;                       // -> array/indexable
  var perCand: [cps.domain] list(int, parSafe=false);
  forall i in cps.domain with (ref perCand, ref state) {
    const (n1, n2) = cps[i];
    if (findingIsos && isFeasible_ISO(n1,n2,state)) || (findingMonos && isFeasible_MONO(n1,n2,state)) {
      var newState = state.clone();
      addToTinTout(n1, n2, newState);
      perCand[i] = recursiveMatchSaverFast(newState, depth + 1);
    }
  }
  for i in cps.domain do for m in perCand[i] do allmappings.pushBack(m);
  return allmappings;
}
```

Then `appendLock`/`appendBlock` can be deleted entirely.

## Correctness + notes
- **Race-free:** no list is shared during any `forall`; each task writes only its own slot. Each
  match's `numSubgraphVertices`-block is produced whole into one slot, so it can't interleave.
- **Order:** the post-`forall` concat is in iteration order, so output ordering is deterministic
  (a nice bonus — the lock version's order was nondeterministic; harmless either way for orbits).
- **Memory:** `perEdge`/`perCand` together hold the same total as the old `solutions` — no extra.
- **Even faster (later):** the sequential concat is O(output); if it ever matters, prefix-sum the
  per-slot sizes and write into a pre-allocated array in parallel (fully parallel gather). Not
  needed first cut.
- **DRAFT:** Bartosz must compile + verify the `candidatePairs` indexability point and the
  `vertexCentric` adaptation. Then re-run the reg strong-scaling sweep to confirm the curve
  extends past 8 cores.
