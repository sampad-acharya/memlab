// numa_bench.cpp -- local vs. remote memory latency and bandwidth.
//
// For every (cpu_node, mem_node) pair we:
//   1. mmap a buffer and mbind() it to `mem_node` BEFORE first touch, so the
//      pages are physically allocated on that node,
//   2. pin the measuring thread to a CPU on `cpu_node`,
//   3. verify page residency with move_pages(),
//   4. measure pointer-chase latency (random cycle, 64B stride, working set >
//      LLC so every access is a DRAM access), and
//   5. measure sequential read bandwidth (single thread, and optionally with N
//      threads on the source node to load the interconnect).
//
// The diagonal of each matrix is local access; off-diagonal is remote.  On a
// single-node machine the cross-node comparison is impossible -- the program
// says so and just reports the local baseline plus a first-touch / page
// residency smoke test.
//
// Only glibc + raw syscalls are used (mbind, move_pages), so there is no
// libnuma / libnuma-dev build dependency.
//
// Usage:
//   ./numa_bench [--size-mb N] [--reps N] [--lat-steps N]
//                [--threads N]   (aggregate bandwidth with N threads/node)
//                [--quick]

#include "common.hpp"

#include <sys/syscall.h>
#include <dirent.h>
#include <cctype>
#include <cerrno>
#include <string>
#include <fstream>
#include <sstream>
#include <thread>
#include <atomic>

using namespace ml;

#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif

// mempolicy modes / flags (from <linux/mempolicy.h>)
static const int  MPOL_BIND       = 2;
static const unsigned MPOL_MF_MOVE   = 1 << 1;
static const unsigned MPOL_MF_STRICT = 1 << 0;

static long sys_mbind(void* addr, unsigned long len, int mode,
                      const unsigned long* nmask, unsigned long maxnode,
                      unsigned flags) {
  return syscall(SYS_mbind, addr, len, mode, nmask, maxnode, flags);
}
static long sys_move_pages(int pid, unsigned long count, void** pages,
                           const int* nodes, int* status, int flags) {
  return syscall(SYS_move_pages, pid, count, pages, nodes, status, flags);
}

// ------------------------------------------------------------- node discovery
struct Node {
  int id = 0;
  std::vector<int> cpus;
};

static std::vector<int> parse_cpulist(const std::string& s) {
  std::vector<int> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (tok.empty()) continue;
    size_t dash = tok.find('-');
    if (dash == std::string::npos) {
      out.push_back(std::stoi(tok));
    } else {
      int a = std::stoi(tok.substr(0, dash));
      int b = std::stoi(tok.substr(dash + 1));
      for (int i = a; i <= b; ++i) out.push_back(i);
    }
  }
  return out;
}

static std::vector<Node> discover_nodes() {
  std::vector<Node> nodes;
  DIR* d = opendir("/sys/devices/system/node");
  if (!d) return nodes;
  for (dirent* e; (e = readdir(d));) {
    if (strncmp(e->d_name, "node", 4) != 0 || !isdigit((unsigned char)e->d_name[4]))
      continue;
    Node n;
    n.id = atoi(e->d_name + 4);
    std::string base = std::string("/sys/devices/system/node/") + e->d_name;
    std::ifstream cl(base + "/cpulist");
    std::string line;
    std::getline(cl, line);
    n.cpus = parse_cpulist(line);
    if (!n.cpus.empty()) nodes.push_back(std::move(n));
  }
  closedir(d);
  std::sort(nodes.begin(), nodes.end(),
            [](const Node& a, const Node& b) { return a.id < b.id; });
  return nodes;
}

// --------------------------------------------------------------- buffer alloc
// mmap `len` bytes bound to `node`, pre-faulted.  Aborts on failure.
static unsigned char* alloc_on_node(size_t len, int node) {
  void* p = mmap(nullptr, len, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) { perror("mmap"); exit(1); }

  if (node < 0 || node >= (int)(8 * sizeof(unsigned long))) {
    fprintf(stderr, "node id %d out of range for this simple bitmask\n", node);
    exit(1);
  }
  unsigned long mask = 1ul << node;
  const unsigned long maxnode = 8 * sizeof(mask);
  if (sys_mbind(p, len, MPOL_BIND, &mask, maxnode, 0) != 0)
    fprintf(stderr, "  mbind(node %d) failed: %s\n", node, strerror(errno));

  if (madvise(p, len, MADV_POPULATE_WRITE) != 0)
    memset(p, 1, len);                           // first touch -> faults on node

  return reinterpret_cast<unsigned char*>(p);
}

// Sample up to `sample` pages and report how many landed on `want`.
static void residency(unsigned char* p, size_t len, int want, size_t sample = 2048) {
  size_t ps = page_size();
  size_t n = std::min(sample, len / ps);
  std::vector<void*> pages(n);
  std::vector<int> st(n, -1);
  for (size_t i = 0; i < n; ++i) pages[i] = p + i * ps;
  if (sys_move_pages(0, n, pages.data(), nullptr, st.data(), 0) != 0) {
    fprintf(stderr, "  move_pages query failed: %s\n", strerror(errno));
    return;
  }
  size_t ok = 0, other = 0, err = 0;
  for (int v : st) (v < 0 ? err : (v == want ? ok : other))++;
  printf("      residency: %zu/%zu pages on node %d", ok, n, want);
  if (other) printf(", %zu elsewhere", other);
  if (err) printf(", %zu unknown", err);
  printf("\n");
}

// ------------------------------------------------------------- bandwidth kernels
// Sequential read: sum of 64-bit words.  -O3 -march=native vectorizes this into
// a wide load + add; integer add is associative so multiple accumulators too.
static double read_bw_gbps(const unsigned char* p, size_t len, int reps) {
  const uint64_t* a = reinterpret_cast<const uint64_t*>(p);
  size_t n = len / sizeof(uint64_t);
  uint64_t acc = 0;
  for (size_t i = 0; i < n; ++i) acc += a[i];        // warm

  uint64_t t0 = now_ns();
  for (int r = 0; r < reps; ++r)
    for (size_t i = 0; i < n; ++i) acc += a[i];
  uint64_t dt = now_ns() - t0;
  sink(acc);
  return double(len) * reps / double(dt);           // bytes/ns == GB/s
}

// N threads pinned to `cpus`, each streaming its own slice; returns aggregate.
static double read_bw_mt_gbps(const unsigned char* p, size_t len,
                              const std::vector<int>& cpus, int nthreads,
                              int reps) {
  nthreads = std::max(1, std::min<int>(nthreads, (int)cpus.size()));
  size_t chunk = (len / nthreads) & ~size_t(63);
  std::vector<double> part(nthreads, 0.0);
  std::vector<std::thread> th;
  std::atomic<int> ready{0};
  std::atomic<bool> go{false};

  for (int t = 0; t < nthreads; ++t) {
    th.emplace_back([&, t] {
      pin_to_cpu(cpus[t]);
      const uint64_t* a =
          reinterpret_cast<const uint64_t*>(p + size_t(t) * chunk);
      size_t n = chunk / sizeof(uint64_t);
      uint64_t acc = 0;
      for (size_t i = 0; i < n; ++i) acc += a[i];    // warm
      ready.fetch_add(1);
      while (!go.load(std::memory_order_acquire)) { }

      uint64_t t0 = now_ns();
      for (int r = 0; r < reps; ++r)
        for (size_t i = 0; i < n; ++i) acc += a[i];
      uint64_t dt = now_ns() - t0;
      sink(acc);
      part[t] = double(chunk) * reps / double(dt);
    });
  }
  while (ready.load() < nthreads) { }
  go.store(true, std::memory_order_release);
  for (auto& x : th) x.join();

  double sum = 0;
  for (double v : part) sum += v;
  return sum;
}

// --------------------------------------------------------------------- tables
static void print_matrix(const char* title, const char* unit,
                         const std::vector<Node>& nodes,
                         const std::vector<std::vector<double>>& m) {
  printf("\n%s  [%s]\n", title, unit);
  printf("            ");
  for (const Node& mn : nodes) printf("  mem@nd%-3d", mn.id);
  printf("\n");
  for (size_t i = 0; i < nodes.size(); ++i) {
    printf("  cpu@nd%-3d ", nodes[i].id);
    for (size_t j = 0; j < nodes.size(); ++j) printf("  %8.2f", m[i][j]);
    printf("\n");
  }
}

int main(int argc, char** argv) {
  size_t size_mb = 256;
  int reps = 4;
  size_t lat_steps = 20'000'000;
  int threads = 0;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto val = [&] { return (i + 1 < argc) ? std::stoull(argv[++i]) : 0ull; };
    if (a == "--size-mb") size_mb = val();
    else if (a == "--reps") reps = int(val());
    else if (a == "--lat-steps") lat_steps = val();
    else if (a == "--threads") threads = int(val());
    else if (a == "--quick") { size_mb = 64; reps = 2; lat_steps = 5'000'000; }
    else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
  }

  const size_t len = size_mb << 20;
  const size_t stride = 64;
  const size_t slots = len / stride;

  std::vector<Node> nodes = discover_nodes();
  if (nodes.empty()) {
    fprintf(stderr, "could not read /sys/devices/system/node\n");
    return 1;
  }

  printf("# NUMA latency / bandwidth microbenchmark\n");
  printf("# nodes detected: %zu\n", nodes.size());
  for (const Node& n : nodes)
    printf("#   node %d: %zu cpus (%d..%d)\n", n.id, n.cpus.size(),
           n.cpus.front(), n.cpus.back());
  printf("# per-cell working set: %zu MiB   latency steps: %zu   bw reps: %d\n",
         size_mb, lat_steps, reps);
  if (threads) printf("# aggregate-bandwidth threads per node: %d\n", threads);

  if (nodes.size() == 1) {
    printf("\n# Only one NUMA node -- local vs. remote cannot be compared on\n"
           "# this machine.  Reporting the local baseline plus a first-touch /\n"
           "# residency smoke test.  Run on a multi-socket box (check with\n"
           "# `numactl --hardware`) to see the full matrices.\n");
  }

  size_t nnodes = nodes.size();
  std::vector<std::vector<double>> lat(nnodes, std::vector<double>(nnodes, 0));
  std::vector<std::vector<double>> bw(nnodes, std::vector<double>(nnodes, 0));
  std::vector<std::vector<double>> bw_mt(nnodes, std::vector<double>(nnodes, 0));

  for (size_t ci = 0; ci < nnodes; ++ci) {
    if (!pin_to_cpu(nodes[ci].cpus.front()))
      fprintf(stderr, "warning: cannot pin to cpu %d\n", nodes[ci].cpus.front());

    for (size_t mi = 0; mi < nnodes; ++mi) {
      printf("\n[cpu node %d] -> [mem node %d]  %s\n", nodes[ci].id, nodes[mi].id,
             ci == mi ? "(local)" : "(remote)");
      unsigned char* buf = alloc_on_node(len, nodes[mi].id);
      residency(buf, len, nodes[mi].id);

      build_random_chain(buf, slots, stride, 0x1234 ^ (ci * 131 + mi));
      double ns = measure_chase(buf, slots,
                                /*target_ms=*/double(lat_steps) / 20000.0);
      lat[ci][mi] = ns;

      double g = read_bw_gbps(buf, len, reps);
      bw[ci][mi] = g;

      double gmt = 0;
      if (threads) {
        gmt = read_bw_mt_gbps(buf, len, nodes[ci].cpus, threads, reps);
        bw_mt[ci][mi] = gmt;
      }

      printf("      latency  : %8.2f ns/access", ns);
#ifdef ML_X86
      printf("  (~%.0f cycles)", ns_to_cycles(ns));
#endif
      printf("\n      read bw  : %8.2f GB/s  (1 thread)\n", g);
      if (threads)
        printf("      read bw  : %8.2f GB/s  (%d threads on node %d)\n", gmt,
               threads, nodes[ci].id);

      munmap(buf, len);
    }
  }

  if (nnodes > 1) {
    print_matrix("Pointer-chase latency", "ns/access", nodes, lat);
    print_matrix("Sequential read bandwidth, 1 thread", "GB/s", nodes, bw);
    if (threads)
      print_matrix("Sequential read bandwidth, N threads/node", "GB/s", nodes,
                   bw_mt);

    printf("\n# local vs remote (averaged over nodes)\n");
    double ll = 0, lr = 0, bl = 0, br = 0;
    int nl = 0, nr = 0;
    for (size_t i = 0; i < nnodes; ++i)
      for (size_t j = 0; j < nnodes; ++j) {
        if (i == j) { ll += lat[i][j]; bl += bw[i][j]; nl++; }
        else        { lr += lat[i][j]; br += bw[i][j]; nr++; }
      }
    if (nl && nr) {
      printf("#   latency : local %.1f ns   remote %.1f ns   (%.2fx)\n",
             ll / nl, lr / nr, (lr / nr) / (ll / nl));
      printf("#   read bw : local %.1f GB/s remote %.1f GB/s (%.2fx)\n",
             bl / nl, br / nr, (br / nr) / (bl / nl));
    }

    // ---- first-touch demonstration -----------------------------------------
    printf("\n# first-touch: allocate WITHOUT binding, touch from node %d,\n"
           "# then check where the pages landed.\n", nodes.back().id);
    void* p = mmap(nullptr, len, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p != MAP_FAILED) {
      std::thread([&] {
        pin_to_cpu(nodes.back().cpus.front());
        memset(p, 1, len);
      }).join();
      residency(reinterpret_cast<unsigned char*>(p), len, nodes.back().id);
      munmap(p, len);
    }
  }

  return 0;
}
