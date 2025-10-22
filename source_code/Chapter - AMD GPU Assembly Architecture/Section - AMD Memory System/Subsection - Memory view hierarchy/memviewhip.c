#include 
#include 

int main() {
    const size_t S = 1<<24;                      // 16 MiB buffer
    void *ptr = NULL;
    int dev = 0;
    hipSetDevice(dev);

    // Allocate managed memory visible to host and device.
    hipHostMalloc(&ptr, S, hipHostMallocMapped); // pinned host-mapped allocation

    // Advise that device should be preferred location for this range.
    // This steers migration and caching behavior in the driver/hardware.
    hipMemAdvise(ptr, S, hipMemAdviseSetPreferredLocation, dev);

    // Optionally mark as read-mostly to reduce CPU write-back traffic.
    hipMemAdvise(ptr, S, hipMemAdviseSetReadMostly, dev);

    // Prefetch to the device to populate page tables and TLBs.
    hipMemPrefetchAsync(ptr, S, dev, 0);         // async prefetch on current stream

    // Kernel launch would follow; here we just synchronize.
    hipDeviceSynchronize();

    hipHostFree(ptr);
    return 0;
}