# ORCA — Provenance

**File:** `orca.cpp` (vendored verbatim, unmodified)
**Upstream:** https://github.com/thocevar/orca (file `orca.cpp`, branch `master`)
**Retrieved:** 2026-06-10 via `curl` (see Task 1 of the Phase 1a plan)

**Authors:** Tomaž Hočevar and Janez Demšar.

**Citation:**
> Tomaž Hočevar, Janez Demšar. "A combinatorial approach to graphlet counting."
> *Bioinformatics* 30(4):559–565, 2014. doi:10.1093/bioinformatics/btt717

**License:** ORCA is distributed by its authors as open source (see the upstream
repository for the exact terms). This file is vendored unmodified for local,
research use as an exact per-vertex graphlet-orbit counter. The compiled binary
`third_party/orca/orca` is built locally and is gitignored.

**Role in this repo:** local orbit-count backend for `src/datasets/orca_orbits.py`
(Phase 1a). When HiPerMotif is connected (Phase LAST), ORCA becomes the correctness
oracle for HiPerMotif's |Aut|-normalized orbit counts.
