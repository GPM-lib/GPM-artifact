#include "tuning_schedules.cuh"
__device__ int linear_search_sm(int neighbor, int* shared_partition,
                                int* partition, int* bin_count, int bin,
                                int BIN_START) {
  for (;;) {
    int i = bin;
    int len = bin_count[i];
    int step = 0;
    int nowlen;
    if (len < SHARED_BUCKET_SIZE)
      nowlen = len;
    else
      nowlen = SHARED_BUCKET_SIZE;
    while (step < nowlen) {
      if (shared_partition[i] == neighbor) {
        return 1;
      }
      i += BLOCK_BUCKET_NUM;
      step += 1;
    }
    len -= SHARED_BUCKET_SIZE;
    i = bin + BIN_START;
    step = 0;
    while (step < len) {
      if (partition[i] == neighbor) {
        return 1;
      }
      i += BLOCK_BUCKET_NUM;
      step += 1;
    }
    if (len + SHARED_BUCKET_SIZE < 99)
      break;
    bin++;
  }
  return 0;
}
template <typename VID, typename VLABEL>
__global__ void tc_hi_warp_vertex(VID nv, dev::Graph<VID, VLABEL> g,
                                  VID* partition, AccType* total, int* G_INDEX,
                                  int block_range) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = thread_id / WARP_SIZE;  // global warp index
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);            // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;      // warp index within the CTA
  int num_warps = WARPS_PER_BLOCK * gridDim.x;  // total number of active warps

  __shared__ int bin_count[BLOCK_BUCKET_NUM];
  __shared__ int shared_partition[BLOCK_BUCKET_NUM * SHARED_BUCKET_SIZE + 1];
  __shared__ int shared_vertex;

  unsigned long long __shared__ G_counter;
  if (threadIdx.x == 0) {
    G_counter = 0;
  }

  int BIN_START = blockIdx.x * BLOCK_BUCKET_NUM * TR_BUCKET_SIZE;
  unsigned long long P_counter = 0;
  int vertex = blockIdx.x;
  // if vertex's degree is large than threshold(USE_CTA), use block to process
  while (vertex < block_range) {
    int degree = g.edge_end(vertex) - g.edge_begin(vertex);
    // Notice: this break need sort graph by degree descending
    if (degree <= USE_CTA)
      break;
    int v_start = g.edge_begin(vertex);
    int v_end = g.edge_end(vertex);
    int now = threadIdx.x + v_start;
    int bucket_module = BLOCK_BUCKET_NUM - 1;
    int binOffset = 0;
    // step1: clean hash bucket
    for (int i = threadIdx.x; i < BLOCK_BUCKET_NUM; i += blockDim.x)
      bin_count[i] = 0;
    __syncthreads();

    HASH_INSERT(threadIdx.x, blockDim.x);
    __syncthreads();
    BLOCK_HASH_LOOKUP();
    __syncthreads();
    BLOCK_NEXT_WORK_CATCH(shared_vertex, vertex, &G_INDEX[1], gridDim.x);
    __syncthreads();
  }
  __syncthreads();
  // //if vertex's degree is small than threshold(USE_CTA), use warp to process
  vertex = block_range + warp_id;
  while (vertex < nv) {
    int v_start = g.edge_begin(vertex);
    int v_end = g.edge_end(vertex);
    int degree = v_end - v_start;
    // Notice: this break need sort graph by degree descending
    if (degree < USE_WARP)
      break;
    int bucket_module = WARP_BUCKET_NUM - 1;
    int binOffset = warp_lane * WARP_BUCKET_NUM;
    // step1: clean hash bucket
    for (int i = binOffset + thread_lane; i < binOffset + WARP_BUCKET_NUM;
         i += WARP_SIZE)
      bin_count[i] = 0;
    __syncwarp();

    // step2: insert v'neighbour u into hash bucket
    //! Notice: use g.N + g.deg + ptr will cause 2-3ms time waste!
    HASH_INSERT(thread_lane, WARP_SIZE);
    __syncwarp();
    WARP_HASH_LOOKUP();
    __syncwarp();
    WARP_NEXT_WORK_CATCH(vertex, &G_INDEX[2], num_warps);
    __syncwarp();
  }

  atomicAdd(&G_counter, P_counter);
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(&total[0], G_counter);
  }
}