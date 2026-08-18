# singularity-eos — Agent Guide (developing it with Spack)

This is Pacific Fusion's **fork** of singularity-eos
(`git@github.com:ScottPFC/singularity-eos.git`), checked out here so we can extend the EOS
models FLASH's mixture (PTE) closures use. The EOS models are **header-only**
(`singularity-eos/eos/*.hpp`), so editing a header is enough to change behavior — but FLASH does
**not** compile against this checkout directly. Read this before editing, or you will rebuild the
wrong thing.

## How FLASH actually links singularity-eos

- singularity-eos is a **Spack package**. Spack fetches it from the fork at a **pinned commit** in
  `IGNITE/pfc/meta/spack/repos/.../singularity_eos/package.py`
  (`package.version("1.11.0", commit=...)`) and installs it under
  `IGNITE/build/spack-installs/…/singularity-eos-1.11.0-<hash>/`.
- FLASH includes from that Spack prefix (`ext/flash/sites/ubuntu/Makefile.h`:
  `SINGULARITY_EOS_PATH = $(IGNITE_PREFIX_DIR)`, `-I$SINGULARITY_EOS_PATH/include`), resolved via the
  Spack env **view** at `IGNITE/pfc/meta/spack/.spack-env/view`.
- **This checkout (`ext/singularity-eos`) is invisible to the build by itself.** So is the stray
  `ext/singularity-eos/build/` + `IGNITE/build/ext/include` — that's an unrelated manual CMake build
  FLASH never includes. Copying headers there does nothing.

⚠️ **A plain `pf-spack install` git-fetches a CLEAN pinned commit into a Spack stage — it drops any
uncommitted edits in this checkout.** That is the "wrong commit / lost edits" trap.

## Build LOCAL (uncommitted) edits into FLASH — `spack develop`

The Spack env lives at `IGNITE/pfc/meta/spack` (direnv-activated; `spack env status` shows
`In environment …/pfc/meta/spack`). Always pass `-e pfc/meta/spack` to be explicit.

```bash
cd $IGNITE_DIR
# 1. Point Spack at THIS working tree (adds a develop: block to pfc/meta/spack/spack.yaml).
#    Spack then builds from the local path, ignoring the pinned commit — no wrong-commit risk.
spack -e pfc/meta/spack develop --path $IGNITE_DIR/ext/singularity-eos singularity-eos@1.11.0

# 2. Re-concretize (dependency solve, ~1.5 min for this env; only singularity re-installs). Once.
bash pfc/meta/spack/concretize.sh

# 3. Rebuild singularity from the local tree (~1 min; header-only models + a small lib/fortran compile).
pf-spack install -j $(nproc)      # == pfc/meta/spack/install.sh -> spack -e pfc/meta/spack install

# 4. Rebuild FLASH against the refreshed view.
cd pfc/sim/problems/<problem> && parflow-set-make -problem <deck>.py -site=ubuntu
```

**Iterate:** edit a header here → `pf-spack install -j $(nproc)` → rebuild FLASH. No re-concretize
needed unless the spec/hash changes. Confirm the view picked up a change with e.g.
`grep -c <NewSymbol> pfc/meta/spack/.spack-env/view/include/singularity-eos/eos/eos_type_lists.hpp`.

### ⚠️ Adding a NEW header needs two extra steps

Editing an existing header is the easy case: the view already symlinks it, so `pf-spack install`
plus a FLASH rebuild is enough. **Adding one silently fails twice**, because with `spack develop`
the spec hash does not change when you edit sources — and the hash is spack's cache key.

```bash
# 1. Add the file to the install list, or `spack install` reports success and ships NOTHING.
#    singularity-eos/CMakeLists.txt  ->  the header list (closure/*.hpp, eos/*.hpp, ...)

# 2. Force the rebuild.  A plain `pf-spack install` prints "[+] already installed" and does
#    nothing, because the hash is unchanged by a source edit.
spack -e pfc/meta/spack install --overwrite -y singularity-eos@1.11.0

# 3. Verify the header actually landed in the INSTALL PREFIX (this step does work):
P=$(readlink -f pfc/meta/spack/.spack-env/view/include/singularity-eos/closure/mixed_cell_models.hpp \
    | sed 's|/include/singularity-eos/closure/.*||')
ls -l $P/include/singularity-eos/closure/ | grep <new-header>
```

**The remaining gap is the VIEW, and it is not solved.**  FLASH includes from the view
(`sites/ubuntu/Makefile.h`: `SINGULARITY_EOS_PATH = $(SPACK_ENV)/.spack-env/view`), which is a
symlink farm into the install prefix generated per spec HASH — and with `spack develop` the hash does
not change when you edit or add sources, so a newly added header has no symlink there even after a
successful `--overwrite` install.  What is actually verified:

- `spack -e pfc/meta/spack env view regenerate` **DOES NOT WORK — tested 2026-08-16, question
  closed.** The earlier note left it open on the theory that a regular file at the target path was
  blocking symlink creation. It is not: with the plain copies DELETED first, regenerate exits 0,
  prints nothing, and leaves the closure directory holding only the headers it already had. The
  new header never appears. With `spack develop` the spec hash does not change when sources do, so
  spack considers the view already correct and does nothing. Deleting the copies first is strictly
  worse than leaving them — it removes headers the build needs and regenerate will not restore.
- **Hand-copying the header into the view does work**, and is the current interim route. Copy from
  the INSTALL PREFIX rather than from this checkout, so what lands in the view is exactly what
  spack built:
  ```bash
  V=pfc/meta/spack/.spack-env/view/include/singularity-eos/closure
  P=$(readlink -f $V/mixed_cell_models.hpp | sed 's|/include/singularity-eos/closure/.*||')
  cp $P/include/singularity-eos/closure/<new-header> $V/
  ```
  It must be REDONE after every edit to that header — otherwise FLASH silently compiles the stale
  copy while your source tree looks correct — and a view rebuild or `pf-spack sync` wipes it.

  ⚠️ **Audit the plain copies before trusting a comparison.** They go stale silently and there is
  no warning. Found on 2026-08-16: the view's `pte_cyclic_rhoe.hpp` was missing the 83-line
  minimiser rescue that had been in the fork since 2026-08-10, so every run in between was made
  against a `cyclic_rhoe` WITHOUT it while the source tree said otherwise. Diff every plain copy
  against the fork before starting a solver comparison:
  ```bash
  for h in $V/*.hpp; do [ -L "$h" ] || diff -q "$h" \
      ext/singularity-eos/singularity-eos/closure/$(basename $h); done
  ```
- The reliable end state is the "Land it" section below: commit the fork, push, bump `commit=` in
  `package.py`. That changes the hash, so the view is regenerated with the new file in it and the
  build reproduces from a fresh clone. Do this before anything ships.

Checking the built binary with `nm` is not a verification: `flash4` is stripped to a few dozen
symbols, so every solver looks absent.  Check the object instead —
`nm -C ext/flash/object_<problem>/build/CMakeFiles/lib_flash4.dir/source/singularity_interface.cxx.o
| grep <Symbol>` — which also confirms a template actually INSTANTIATED, something `-fsyntax-only`
on the header never checks.

## Land it (reproducible end state)

```bash
spack -e pfc/meta/spack undevelop singularity-eos      # remove the develop block
```
Then commit the edits to the fork, push, bump `commit=` in `singularity_eos/package.py`, and
`pf-spack install -j $(nproc)`. Now a fresh clone reproduces the build without the develop entry.

## Adding a new EOS model to the variant

Header-only models are registered in `singularity-eos/eos/`:
1. Write `eos_<name>.hpp` (`class Foo : public EosBase<Foo>`). Implement the **full** method set the
   variant dispatches — mirror an existing sibling (e.g. `eos_table_pt.hpp`); a missing method fails
   the compile for every EOS type.
2. `#include` it in `eos_models.hpp`.
3. Add the type to `full_eos_list` in `eos_type_lists.hpp` (under the right feature `#ifdef`).
4. Add the header to `CMakeLists.txt`'s install list (so `spack install` copies it).
5. FLASH-side factory + Fortran binding live in
   `IGNITE/ext/flash/source/physics/Eos/EosMain/multiTemp/Multitype/Singularity/`
   (`singularity_interface.{hxx,cxx}`, `singularity_fortran_interface.F90`).

## PF-specific models here (context)

- `eos_table_pt.hpp` — `TableDependsPT`: a pre-inverted **(P,T)** table for the cyclic PTE closure
  (`PTESolverPTCyclic`). Direct O(1) interpolation; only spans the monotone, positive-P region, so a
  cold condensed material's low-P shoulder is truncated.
- `eos_table_rhot.hpp` — `TableDependsRhoT`: answers the same (P,T) primitives by a **live
  dense-branch density root on the full (rho,T) min-F envelope** (port of
  `IGNITE/pfc/sim/matdata/_pte_rhot.py::RhoTPteEos`). Full-domain, no truncation; `MinimumPressure()=0`
  keeps the solver out of tension. Selected by the FLASH runtime param `eos_ptCyclicRhoT`. The
  envelope table is written by `IGNITE/designs/append_rhot_envelope.py` (appends a `dependsRhoT` group
  to the ion `_pt.sp5`).
