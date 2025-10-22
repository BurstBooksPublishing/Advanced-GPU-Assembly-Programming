#include 

// Simple BVH node struct (fits many production BVH layouts).
struct BVHNode { float bbox[6]; int left; int right; int triIdx; };

// Allocate managed BVH and prefetch to GPU; then launch ray kernel.
int main() {
  const size_t numNodes = 4<<20;                     // large BVH
  BVHNode *bvh;
  cudaMallocManaged(&bvh, numNodes * sizeof(BVHNode)); // unified memory

  // Populate BVH on CPU (omitted)...
  // Advise that BVH is read-mostly and prefer GPU 0.
  cudaMemAdvise(bvh, numNodes*sizeof(BVHNode), cudaMemAdviseSetReadMostly, 0);
  cudaMemAdvise(bvh, numNodes*sizeof(BVHNode), cudaMemAdviseSetPreferredLocation, 0);

  // Create a stream and prefetch the BVH to GPU asynchronously.
  cudaStream_t s;
  cudaStreamCreate(&s);
  cudaMemPrefetchAsync(bvh, numNodes*sizeof(BVHNode), 0, s); // pre-stage to GPU 0

  // Launch ray kernel only after prefetch (or overlap prefetch with other work).
  // rayKernel<<>>(bvh, ...); // kernel uses BVH for traversal

  cudaStreamSynchronize(s); // ensure pages migrated before heavy traversal
  cudaStreamDestroy(s);
  cudaFree(bvh);
  return 0;
}