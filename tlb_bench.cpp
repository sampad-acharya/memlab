// tlb_bench.cpp -- isolate the cost of a DTLB miss / page-table walk.
//
// Idea
// ----
// Touch exactly one 64-byte line per page, and pointer-chase through those lines
// in a random cycle.  Sweep the number of pages `P`:
//
//   * The DATA actually touched is  P * 64 bytes  -- stays inside L2/L3 for the
//     whole sweep on a modern CPU, so cache misses are NOT what we are measuring.
//   * The ADDRESS SPAN walked is    P * 4096 bytes -- grows 64x faster, so it
//     blows past the L1 DTLB (tens of entries) and then the L2 STLB (a couple
//     thousand entries) long before the data leaves cache.
//
// So the curve of ns/access vs. P shows plateaus:
//   P small ............ working set fits L1 DTLB           -> ~L1d hit latency
//   P medium ........... fits L2 STLB but not L1 DTLB       -> + STLB latency
//   P large ............ exceeds STLB reach                 -> + full page walk
//
// Three configurations are run per P:
//   4K/random  the headline TLB curve (unpredictable page stream)
//   4K/seq     sequential page order -- shows the page-walk / next-line
//              prefetchers hiding a big chunk of the cost
//   2M/random  transparent huge pages: 512x more TLB reach per entry, so the
//              random curve stays flat far longer
//
// Usage: ./tlb_bench [--cpu N] [--max-pages N] [--seq-only] [--no-huge]

#include "common.hpp"

#include <cinttypes>
#include <cctype>
#include <string>

using namespace ml;

#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif

static const size_t HUGE = 2ull << 20;   // 2 MiB
static const size_t LINE = 64;           // one cache line touched per page

// Slot pitch.  We deliberately use PAGE + LINE, not PAGE:
//   * still exactly one fresh page per access (pitch > page size), so the TLB
//     behaviour we want to measure is unchanged, but
//   * the touched cache-line addresses now advance by (PAGE+LINE)/LINE = 65
//     cache sets per step.  65 is coprime with the L1d/L2 set count (a power of
//     two), so the lines spread evenly across every set instead of colliding in
//     a handful of them -- a plain PAGE pitch aliases into 64 sets and produces
//     conflict misses that swamp the TLB signal.
static size_t slot_pitch() { return size_t(ml::page_size()) + LINE; }

// --------------------------------------------------- huge-page introspection
// Return how many bytes of [addr, addr+len) are backed by transparent huge
// pages, by parsing /proc/self/smaps for the VMA containing `addr`.
static long anon_hugepage_bytes(void* addr) {
  FILE* f = fopen("/proc/self/smaps", "r");
  if (!f) return -1;
  char line[512];
  uintptr_t a = reinterpret_cast<uintptr_t>(addr);
  bool in_vma = false;
  long kb = -1;
  while (fgets(line, sizeof line, f)) {
    uintptr_t s, e;
    if (sscanf(line, "%" SCNxPTR "-%" SCNxPTR " ", &s, &e) == 2) {
      in_vma = (a >= s && a < e);
    } else if (in_vma && strncmp(line, "AnonHugePages:", 14) == 0) {
      long v = 0;
      sscanf(line + 14, "%ld", &v);
      kb = v;
      break;
    }
  }
  fclose(f);
  return kb < 0 ? kb : kb * 1024;
}

// --------------------------------------------------------------- region alloc
// Map `span` bytes, optionally asking for THP, and fault every page in now so
// timing never pays a minor page fault.
static unsigned char* map_region(size_t span, bool want_huge) {
  size_t align = want_huge ? HUGE : size_t(page_size());
  span = (span + align - 1) & ~(align - 1);

  void* p = mmap(nullptr, span + align, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) { perror("mmap"); exit(1); }

  // align base up so a 2M-backed region starts on a 2M boundary (helps THP)
  uintptr_t base = (reinterpret_cast<uintptr_t>(p) + align - 1) & ~(align - 1);
  unsigned char* r = reinterpret_cast<unsigned char*>(base);

  if (want_huge)
    madvise(r, span, MADV_HUGEPAGE);
  else
    madvise(r, span, MADV_NOHUGEPAGE);

  if (madvise(r, span, MADV_POPULATE_WRITE) != 0) {   // pre-fault (Linux >= 5.14)
    for (size_t off = 0; off < span; off += page_size())
      r[off] = 0;
  }
  return r;
}

int main(int argc, char** argv) {
  int cpu = 2;
  size_t max_pages = 1u << 17;   // 512 MiB span, 8 MiB data
  bool do_huge = true;
  bool seq_only = false;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto val = [&]() { return (i + 1 < argc) ? std::stoull(argv[++i]) : 0ull; };
    if (a == "--cpu") cpu = int(val());
    else if (a == "--max-pages") max_pages = val();
    else if (a == "--no-huge") do_huge = false;
    else if (a == "--seq-only") seq_only = true;
    else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
  }

  if (!pin_to_cpu(cpu))
    fprintf(stderr, "warning: could not pin to cpu %d\n", cpu);

  const size_t PS = page_size();
  const size_t PITCH = slot_pitch();

  // geometric-ish sweep with extra detail near the DTLB/STLB boundaries
  std::vector<size_t> sweep = {
      16,   24,   32,   48,   64,   80,   96,   112,  128,   160,   192,
      256,  384,  512,  768,  1024, 1536, 2048, 3072, 4096,  6144,  8192,
      12288, 16384, 24576, 32768, 49152, 65536, 98304, 131072, 262144, 524288};
  while (!sweep.empty() && sweep.back() > max_pages) sweep.pop_back();

  printf("# TLB pressure microbenchmark\n");
  printf("# cpu=%d  base-page=%zuB  huge-page=%zuKiB  line/page=%zuB\n",
         cpu, PS, HUGE / 1024, LINE);
#ifdef ML_X86
  printf("# TSC ~ %.3f GHz (calibrated)\n", tsc_hz() / 1e9);
#endif
  printf("#\n");
  printf("# %-9s %-10s %-10s | %11s %11s %11s   %6s\n", "pages", "span",
         "data", "4K random", "4K seq", "2M random", "THP");
  printf("# %-9s %-10s %-10s | %11s %11s %11s   %6s\n", "", "", "",
         "ns/acc", "ns/acc", "ns/acc", "%span");
  printf("# ------------------------------------------------------------"
         "-------------------------\n");

  auto human = [](size_t bytes, char* out) {
    const char* u[] = {"B", "K", "M", "G"};
    double v = double(bytes);
    int k = 0;
    while (v >= 1024.0 && k < 3) { v /= 1024.0; ++k; }
    snprintf(out, 16, "%.0f%s", v, u[k]);
  };

  double first_4k = 0, last_4k_rand = 0, last_4k_seq = 0, last_2m = 0;
  // cleanest TLB-only delta: the row whose data footprint is still L2-resident
  // (~512 KiB) so 4K vs 2M differ ONLY in page size, not in data cache level.
  double l2_4k = 0, l2_2m = 0;
  size_t l2_np = 0;

  for (size_t np : sweep) {
    size_t span = np * PITCH;
    size_t data = np * LINE;
    char sbuf[16], dbuf[16];
    human(span, sbuf);
    human(data, dbuf);

    // ---- 4K / random ------------------------------------------------------
    double ns_rand = 0;
    {
      unsigned char* r = map_region(span, /*huge=*/false);
      build_random_chain(r, np, PITCH, 0xC0FFEE ^ np);
      ns_rand = measure_chase(r, np);
      munmap(r, span);
    }

    // ---- 4K / sequential ------------------------------------------------
    double ns_seq = 0;
    {
      unsigned char* r = map_region(span, /*huge=*/false);
      build_seq_chain(r, np, PITCH);
      ns_seq = measure_chase(r, np);
      munmap(r, span);
    }

    // ---- 2M / random --------------------------------------------------
    double ns_huge = 0;
    double huge_pct = 0;
    if (do_huge && !seq_only) {
      size_t hspan = (span + HUGE - 1) & ~(HUGE - 1);
      unsigned char* r = map_region(hspan, /*huge=*/true);
      build_random_chain(r, np, PITCH, 0xBEEF ^ np);   // still 1 fresh page/access
      ns_huge = measure_chase(r, np);
      long hb = anon_hugepage_bytes(r);
      if (hb > 0) huge_pct = 100.0 * double(hb) / double(hspan);
      munmap(r, hspan);
    }

    printf("  %-9zu %-10s %-10s | %11.2f %11.2f %11.2f   %5.0f%%\n", np, sbuf,
           dbuf, ns_rand, ns_seq, (do_huge && !seq_only) ? ns_huge : 0.0,
           huge_pct);
    fflush(stdout);

    if (first_4k == 0) first_4k = ns_rand;
    last_4k_rand = ns_rand;
    last_4k_seq = ns_seq;
    if (ns_huge) last_2m = ns_huge;
    if (data <= (768u << 10)) { l2_4k = ns_rand; l2_2m = ns_huge; l2_np = np; }
  }

  printf("\n# summary\n");
  printf("#   L1-DTLB + L1d resident (smallest sweep)   : %8.2f ns\n", first_4k);
  printf("#   beyond STLB, 4K random                    : %8.2f ns\n",
         last_4k_rand);
  printf("#   beyond STLB, 4K sequential                : %8.2f ns  "
         "(page-walk/next-line prefetch hides %.0f%%)\n",
         last_4k_seq,
         last_4k_rand > first_4k
             ? 100.0 * (last_4k_rand - last_4k_seq) / (last_4k_rand - first_4k)
             : 0.0);
  if (last_2m)
    printf("#   beyond STLB, same span, 2M pages, random  : %8.2f ns\n", last_2m);
  printf("#\n");
  printf("# Isolated TLB / page-walk cost = (4K random - 2M random) at matched\n"
         "# span & data footprint (only the page size differs):\n");
  if (l2_2m > 0) {
    double d = l2_4k - l2_2m;
    printf("#   at %zu pages  (data ~L2-resident, PTEs cached) : %6.2f ns", l2_np,
           d);
#ifdef ML_X86
    printf("  (~%.0f cyc)", ns_to_cycles(d));
#endif
    printf("\n");
  }
  if (last_2m > 0) {
    double d = last_4k_rand - last_2m;
    printf("#   beyond STLB (widest sweep)                     : %6.2f ns", d);
#ifdef ML_X86
    printf("  (~%.0f cyc)", ns_to_cycles(d));
#endif
    printf("\n");
  }
  return 0;
}
