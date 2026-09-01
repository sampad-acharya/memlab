// common.hpp -- shared helpers for the memory-layout / cache-behavior microbenchmarks.
//
// Nothing here is exotic: a monotonic clock, an x86 TSC reader with calibration,
// CPU pinning, a small PRNG, an "optimizer fence" sink, and a pointer-chasing
// primitive.  The pointer chase is the workhorse for both the TLB and the NUMA
// latency tests: a single random Hamiltonian cycle through a set of equally
// spaced slots, so every dereference depends on the previous one (no MLP, no
// prefetch) and the address stream is unpredictable to the hardware.
#pragma once

#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#include <algorithm>

#include <sched.h>
#include <unistd.h>
#include <sys/mman.h>

#if defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#define ML_X86 1
#endif

namespace ml {

// ---------------------------------------------------------------- basic timing
inline long page_size() {
  static long p = sysconf(_SC_PAGESIZE);
  return p;
}

inline uint64_t now_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + ts.tv_nsec;
}

// ------------------------------------------------------------------- x86 TSC
#ifdef ML_X86
inline uint64_t tsc_now() {
  unsigned aux;
  _mm_lfence();
  uint64_t t = __rdtscp(&aux);   // rdtscp waits for prior loads; lfence pins it
  _mm_lfence();
  return t;
}

// Calibrate the TSC against CLOCK_MONOTONIC once, lazily.  On this class of CPU
// the TSC is invariant (constant_tsc / nonstop_tsc), so a single fit is stable.
inline double tsc_hz() {
  static double hz = [] {
    double best = 0.0;
    for (int i = 0; i < 5; ++i) {
      uint64_t n0 = now_ns(), c0 = tsc_now();
      while (now_ns() - n0 < 25000000ull) { /* spin ~25 ms */ }
      uint64_t n1 = now_ns(), c1 = tsc_now();
      double h = double(c1 - c0) * 1e9 / double(n1 - n0);
      if (h > best) best = h;   // take the run with least scheduler interference
    }
    return best;
  }();
  return hz;
}

inline double ns_to_cycles(double ns) { return ns * tsc_hz() / 1e9; }
#endif  // ML_X86

// ---------------------------------------------------------------- CPU pinning
inline bool pin_to_cpu(int cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  return sched_setaffinity(0, sizeof(set), &set) == 0;
}

// ---------------------------------------------------------------------- PRNG
struct Xorshift64 {
  uint64_t s;
  explicit Xorshift64(uint64_t seed = 0x9E3779B97F4A7C15ull) : s(seed ? seed : 1) {}
  uint64_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
  uint64_t below(uint64_t n) { return next() % n; }
};

// ------------------------------------------------------- anti-optimization sink
// Force the compiler to treat `v` as observed, without emitting any code.
template <class T>
inline void sink(const T& v) {
#if defined(__GNUC__)
  asm volatile("" : : "r,m"(v) : "memory");
#else
  volatile T tmp = v;
  (void)tmp;
#endif
}

// --------------------------------------------------------------- pointer chase
// Link `nslots` slots (each `stride` bytes apart, first slot at `base`) into ONE
// random cycle: the first 8 bytes of each slot hold the address of the next.
// The cycle is guaranteed to start at `base` (slot 0), so callers chase from it.
inline void build_random_chain(unsigned char* base, size_t nslots, size_t stride,
                               uint64_t seed) {
  std::vector<size_t> idx(nslots);
  for (size_t i = 0; i < nslots; ++i) idx[i] = i;

  Xorshift64 rng(seed);
  for (size_t i = nslots - 1; i > 0; --i)          // Fisher-Yates
    std::swap(idx[i], idx[rng.below(i + 1)]);

  for (size_t i = 0; i < nslots; ++i)              // pin the cycle start to slot 0
    if (idx[i] == 0) { std::swap(idx[0], idx[i]); break; }

  for (size_t i = 0; i < nslots; ++i) {
    unsigned char* cur = base + idx[i] * stride;
    unsigned char* nxt = base + idx[(i + 1) % nslots] * stride;
    *reinterpret_cast<void**>(cur) = nxt;
  }
}

// Sequential variant: slot i -> slot i+1.  Same footprint, fully predictable
// address stream -- isolates the hardware page-walk prefetcher / next-line
// prefetch from the pure random-access TLB cost.
inline void build_seq_chain(unsigned char* base, size_t nslots, size_t stride) {
  for (size_t i = 0; i < nslots; ++i)
    *reinterpret_cast<void**>(base + i * stride) =
        base + ((i + 1) % nslots) * stride;
}

inline void* chase(void* start, size_t steps) {
  void* p = start;
  while (steps--) p = *reinterpret_cast<void**>(p);
  return p;
}

// Chase for roughly `target_ms` and return ns per dereference.
inline double measure_chase(void* start, size_t nslots, double target_ms = 200.0) {
  void* p = chase(start, nslots);           // warm the structure once
  sink(p);

  uint64_t t0 = now_ns();
  p = chase(start, nslots);                 // probe: time one lap
  uint64_t probe = now_ns() - t0;
  sink(p);
  if (probe == 0) probe = 1;

  size_t laps = std::max<size_t>(1, size_t(target_ms * 1e6 / double(probe)));
  size_t steps = laps * nslots;

  t0 = now_ns();
  p = chase(start, steps);
  uint64_t dt = now_ns() - t0;
  sink(p);
  return double(dt) / double(steps);
}

}  // namespace ml
