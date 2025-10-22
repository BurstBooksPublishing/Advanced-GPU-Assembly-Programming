#include 

__device__ unsigned long long atomicAdd64_cas(unsigned long long* addr,
                                              unsigned long long val) {
  // Load initial value then attempt CAS until success.
  unsigned long long old = *addr;          // read (may be stale)
  unsigned long long assumed;
  do {
    assumed = old;                          // value we think is current
    unsigned long long newval = assumed + val; // compute desired result
    // atomicCAS returns the previous value at *addr
    old = atomicCAS(addr, assumed, newval); // try to swap if unchanged
    // if old == assumed then swap succeeded; else old holds new value
  } while (assumed != old);                 // loop until we win
  return old;                              // original value before add
}