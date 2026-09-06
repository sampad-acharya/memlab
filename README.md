# memlab — TLB & NUMA microbenchmarks

Two small C++17 programs that measure two of the memory-subsystem effects from
the "memory layout and cache behavior" topic:

| program       | what it isolates                                              |
|---------------|--------------------------------------------------------------|
| `tlb_bench`   | cost of a DTLB miss / hardware page-table walk, and how huge pages and access-pattern predictability change it |
| `tlb_latency` | compact version of the same idea: a fixed 4-point page-count sweep, random pointer chase, ns/access only |
| `numa_bench`  | local vs. remote DRAM latency and bandwidth, first-touch placement, page residency |

No third-party libraries. `numa_bench` talks to the kernel directly via the
`mbind(2)` and `move_pages(2)` syscalls, so there is **no `libnuma` /
`libnuma-dev` build dependency**.

## Build & run

```sh
make
./tlb_bench            # ~20 s
./tlb_latency          # ~10 s, no flags
./numa_bench           # ~5 s single-node, longer on a real multi-node box
./numa_bench --threads 8
```

Requires: Linux ≥ 5.14 (for `MADV_POPULATE_WRITE`; there is a fallback),
x86-64 or ARM64, g++/clang++ with `-march=native`.

Pin results down: run under a fixed governor and, ideally, on an isolated core.

```sh
sudo cpupower frequency-set -g performance
taskset -c 2 ./tlb_bench --cpu 2
```

---

## tlb_bench

### Method

Touch exactly **one 64-byte line per page** and pointer-chase through those
lines in a single random cycle (each load depends on the previous one — no
memory-level parallelism, no prefetch). Then sweep the number of pages `P`:

* **data actually touched** = `P × 64 B` — stays inside L2/L3 for most of the
  sweep, so cache-capacity misses are *not* the thing being measured;
* **address span walked** = `P × ~4 KiB` — grows 64× faster, so it overruns the
  L1 DTLB (tens of entries) and then the L2 STLB (a few thousand entries) long
  before the data spills out of cache.

The slot pitch is `PAGE + 64`, not `PAGE`, on purpose: a pitch of exactly one
page aliases every touched line into just 64 cache sets and the resulting
conflict misses drown the TLB signal. `PAGE + 64` advances 65 sets per step,
coprime with the (power-of-two) set count, so lines spread across all sets.

Three configurations per `P`:

| column      | pattern                | shows |
|-------------|------------------------|-------|
| `4K random` | random page order      | the headline TLB curve |
| `4K seq`    | sequential page order  | how much the page-walk / next-line prefetchers hide |
| `2M random` | THP-backed, random     | 512× more reach per TLB entry → curve stays flat far longer |

### Reading the output

`ns/acc` vs. `P` shows plateaus. On the dev box (Intel Core Ultra 9 275HX,
P-core: 48 KiB L1d, ~3 MiB L2; 36 MiB shared L3):

```
P ≤ ~128      ~0.8 ns   working set fits L1 DTLB and L1d
P ~160–768    ~2.1 ns   spilled L1 DTLB → STLB hit         (2M column still ~0.8 ns)
P ~1k–24k     3–10 ns   STLB + growing page-walk-cache pressure
P ≥ ~49k      ~22 ns    past STLB reach AND data now L3-resident
```

The clean, apples-to-apples TLB number is **`4K random − 2M random` at the same
`P`** (identical span and data footprint; only the page size differs):

```
Isolated TLB / page-walk cost:
  data ~L2-resident, PTEs cached : ~5 ns  (~16 cycles)
  beyond STLB                    : ~6 ns  (~20 cycles)
```

i.e. a page walk whose page-table entries are themselves cache-resident costs
~15–20 cycles here. `4K seq` recovers ~70 % of the gap — the walker prefetches
down predictable streams. On a machine under memory pressure, where PTEs get
evicted to DRAM, the same walk can cost 100 ns+.

### Flags

```
--cpu N          core to pin to (default 2)
--max-pages N    largest P in the sweep (default 131072 → ~520 MiB span)
--no-huge        skip the 2M column (e.g. THP disabled)
--seq-only       only the 4K sequential column
```

If the `THP %span` column shows less than 100 %, transparent huge pages didn't
fully back the region (`cat /sys/kernel/mm/transparent_hugepage/enabled`; needs
`always` or `madvise`) and the `2M random` column is not meaningful.

---

## tlb_latency

A minimal companion to `tlb_bench`: same core experiment (one `Node` per 4 KiB
page, random circular pointer chain, dependent-load chase), but no CPU pinning,
no TSC calibration, and no `4K seq` / `2M huge` comparison columns. It just runs
four hard-coded page counts and prints ns per access.

```
32 pages    (0.125 MB)   working set fits the L1 DTLB
256 pages   (1 MB)       spilled to the L2 STLB
2048 pages  (8 MB)       past STLB reach -> page-table walks
16384 pages (64 MB)      severe TLB thrashing (and data now past L2)
```

Because the touched footprint here is `pages x 4 KiB` (a full page each, not one
line per page), the largest sweep also pushes the data itself out of cache, so
its ns/access folds in cache-miss cost on top of the page walk — `tlb_bench` is
the one to use when you need the TLB cost in isolation. No flags.

---

## numa_bench

### Method

For every `(cpu_node, mem_node)` pair:

1. `mmap` a buffer and `mbind(MPOL_BIND)` it to `mem_node` **before first
   touch**, so the physical pages are allocated on that node;
2. pin the measuring thread to a CPU on `cpu_node`;
3. verify placement with `move_pages(2)` (query mode) — printed as `residency:`;
4. **latency** — random pointer chase, 64 B stride, working set (default
   256 MiB) well past the 36 MiB L3, so every access is a DRAM access;
5. **bandwidth** — sequential 64-bit read stream, single thread; with
   `--threads N`, `N` threads pinned across `cpu_node` to load the interconnect.

The diagonal of each matrix is local access; off-diagonal is remote.

### On a multi-node machine you get

```
Pointer-chase latency  [ns/access]
              mem@nd0    mem@nd1
  cpu@nd0       ~85       ~140      ← remote ≈ 1.5–2.0× local
  cpu@nd1      ~140       ~85

Sequential read bandwidth, 1 thread  [GB/s]
  ...                               ← remote ≈ 0.5–0.7× local; the gap widens
                                      with --threads as the link saturates
```

plus a **first-touch** demonstration: a buffer allocated with no binding, then
written from a thread on the last node, ends up with its pages on that node —
placement follows the first writer, not the allocator.

### On a single-node machine (like this dev box)

Cross-node comparison is impossible; the program says so and reports the local
baseline plus the residency smoke test (which still exercises the full
`mbind` + `move_pages` path). Check your topology with `numactl --hardware`.

### Flags

```
--size-mb N    per-cell working set, MiB (default 256)
--reps N       bandwidth repetitions (default 4)
--lat-steps N  pointer-chase steps ≈ measurement time (default 20e6)
--threads N    also measure aggregate bandwidth with N threads per node
--quick        small/fast settings for a smoke test
```

---

## x86-64 vs. ARM64 — why these numbers move

### Pages & TLB reach
* **x86-64**: 4 KiB base page, 2 MiB and 1 GiB huge pages. 4-level (48-bit) or
  5-level paging. L1 DTLB ~64–128 entries; L2 STLB ~1.5–3 K entries → a few MiB
  of 4 KiB reach, hundreds of MiB with 2 MiB pages. Dedicated page-walk caches
  for the upper levels; `PCID` tags TLB entries so context switches don't flush.
* **ARM64**: base page is a **build/runtime choice — 4 KiB, 16 KiB, or 64 KiB**
  (Apple platforms use 16 KiB; Linux servers are usually 4 KiB, sometimes
  64 KiB). Bigger base pages = bigger TLB reach and shallower walks, at the cost
  of internal fragmentation. "Contiguous PTE" hints let 16 entries share one TLB
  slot; block mappings give 2 MiB / 512 MiB / 1 GiB (at 4 KiB granule).
  `TTBR0/TTBR1` split user/kernel translation, and `ASID`s (like `PCID`) avoid
  flush-on-switch. Apple M-series has an unusually large L2 TLB.

Consequence for `tlb_bench`: the plateau boundaries sit at different `P`, and on
a 16 KiB-page ARM64 box the 4 KiB→STLB knee is pushed ~4× further out.

### Struct layout & alignment
* Both are LP64 with natural alignment and `alignof(max_align_t) == 16`
  (`__int128`, SIMD types). Rules for `struct` padding are the same; a layout
  tuned on one is generally fine on the other.
* **Unaligned access**: x86-64 tolerates unaligned scalar loads/stores cheaply
  (still splits across a cache line / page). ARM64 allows unaligned normal
  loads/stores but **not** for atomics or some load/store-pair forms, and a
  misaligned atomic faults — packed structs with atomic fields are a portability
  trap.
* **Cache line / false sharing**: 64 B on mainstream x86-64 and most ARM64
  server parts, but **Apple M-series uses 128 B lines**. `alignas(64)` padding
  that stops false sharing on a PC is insufficient there — prefer
  `std::hardware_destructive_interference_size` (or a 128 B constant) for
  portable padding. `MESI`/`MOESI` coherence details also differ, which shows up
  in `numa_bench --threads` and in any false-sharing test.

### Memory ordering (not exercised here, but same family of effects)
x86-64 is TSO (strong); ARM64 is weakly ordered and needs explicit acquire/
release fences. Lock-free structures that "work" on x86-64 often expose missing
barriers on ARM64.
