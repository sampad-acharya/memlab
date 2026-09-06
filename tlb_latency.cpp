// tlb_latency.cpp -- compact DTLB-latency sweep.
//
// A stripped-down companion to tlb_bench.cpp: one Node at the start of every
// 4 KiB page, linked into a random circular pointer chain, then chased for a
// fixed 100M dereferences.  Sweeping the page count walks the working set from
// "fits in the L1 DTLB" up to "every access is a page-table walk", and the
// printed ns/access shows the plateaus.
//
// Unlike tlb_bench.cpp this does not pin to a core, calibrate the TSC, or run
// the 4K-seq / 2M-huge comparison columns -- it is the minimal version of the
// experiment.  Build: make tlb_latency   (or see the Makefile).

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <random>

constexpr size_t PAGE_SIZE = 4096; // 4 KB standard page size
constexpr size_t ITERATIONS = 100'000'000;

// Node structure aligned to avoid sub-page cache noise
struct Node {
    Node* next;
    char padding[64 - sizeof(Node*)]; // Pad to standard cacheline size (64 bytes)
};

void run_tlb_benchmark(size_t num_pages, const std::string& label) {
    // Allocate memory aligned to 4KB boundary
    size_t allocation_size = num_pages * PAGE_SIZE;
    char* raw_memory = static_cast<char*>(
        #if defined(_MSC_VER)
            _aligned_malloc(allocation_size, PAGE_SIZE)
        #else
            std::aligned_alloc(PAGE_SIZE, allocation_size)
        #endif
    );

    if (!raw_memory) {
        std::cerr << "Memory allocation failed!\n";
        return;
    }

    // Place one Node at the start of each 4KB page
    std::vector<Node*> nodes(num_pages);
    for (size_t i = 0; i < num_pages; ++i) {
        nodes[i] = reinterpret_cast<Node*>(raw_memory + (i * PAGE_SIZE));
    }

    // Randomize node traversal order to defeat CPU hardware prefetchers
    std::mt19937 g(1337);
    std::shuffle(nodes.begin(), nodes.end(), g);

    // Link nodes together in a circular pointer chain
    for (size_t i = 0; i < num_pages; ++i) {
        nodes[i]->next = nodes[(i + 1) % num_pages];
    }

    // Warm up the caches
    Node* current = nodes[0];
    for (size_t i = 0; i < num_pages * 2; ++i) {
        current = current->next;
    }

    // Benchmark the pointer-chasing loop
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < ITERATIONS; ++i) {
        current = current->next; // Forces TLB lookup per access
    }

    auto end = std::chrono::high_resolution_clock::now();

    // Prevent compiler from optimizing away the loop
    if (current == nullptr) std::cout << "Unreachable guard\n";

    std::chrono::duration<double, std::nano> total_ns = end - start;
    double ns_per_access = total_ns.count() / ITERATIONS;

    std::cout << "[" << label << "]\n"
              << "  Pages Touched: " << num_pages
              << " (" << (num_pages * 4) / 1024.0 << " MB working set)\n"
              << "  Avg Latency per Memory Access: " << ns_per_access << " ns\n\n";

    #if defined(_MSC_VER)
        _aligned_free(raw_memory);
    #else
        std::free(raw_memory);
    #endif
}

int main() {
    std::cout << "--- TLB Latency Benchmark ---\n\n";

    // Scenario 1: Small working set (fits comfortably inside L1/L2 TLB)
    run_tlb_benchmark(32, "L1 D-TLB Hit Range");
    run_tlb_benchmark(256, "L2 D-TLB Hit Range");

    // Scenario 2: Exceeding hardware TLB capacity (forces Page Table Walks)
    run_tlb_benchmark(2048, "L2 D-TLB Spill Over");
    run_tlb_benchmark(16384, "Severe TLB Thrashing (64 MB Memory)");

    return 0;
}
