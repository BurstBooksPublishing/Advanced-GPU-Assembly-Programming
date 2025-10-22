// Producer: write data, ensure device visibility, then publish flag
__device__ void producer(int *data, int *flag) {
  data[0] = 42;                 // plain store to global memory
  __threadfence();             // ensure store is visible device-wide
  atomicExch(flag, 1);         // publish: atomic write to flag
}

// Consumer: wait until flag visible, then read data
__device__ void consumer(int *data, int *flag) {
  while (atomicAdd(flag, 0) == 0) { /* spin; atomic read ensures sync */ }
  int v = data[0];              // safe: happens-after relation holds
  // use v...
}