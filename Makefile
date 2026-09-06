CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++17 -march=native -Wall -Wextra -Wno-unused-result
LDFLAGS  ?=

BINS = tlb_bench tlb_latency numa_bench

all: $(BINS)

tlb_bench: tlb_bench.cpp common.hpp
	$(CXX) $(CXXFLAGS) -o $@ tlb_bench.cpp $(LDFLAGS)

tlb_latency: tlb_latency.cpp
	$(CXX) $(CXXFLAGS) -o $@ tlb_latency.cpp $(LDFLAGS)

numa_bench: numa_bench.cpp common.hpp
	$(CXX) $(CXXFLAGS) -pthread -o $@ numa_bench.cpp $(LDFLAGS)

run: all
	./tlb_bench
	./tlb_latency
	./numa_bench

clean:
	rm -f $(BINS)

.PHONY: all run clean
