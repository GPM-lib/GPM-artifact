#ifndef GRAPH_SET_DIFFERENCE_H
#define GRAPH_SET_DIFFERENCE_H
#include "src/graph/search.cuh"

// compute set difference: a - b
template <typename T>
DEV_INLINE T difference_num_bs(T* a, T size_a, T* b, T size_b) {
  // if (size_a == 0) return 0;
  // assert(size_b != 0);
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);  // thread index within the warp
  T num = 0;
  for (auto i = thread_lane; i < size_a; i += WARP_SIZE) {
    auto key = a[i];
    if (!binary_search(b, key, size_b))
      num += 1;
  }
  return num;
}

template <typename T>
DEV_INLINE T difference_num_bs_cache(T* a, T size_a, T* b, T size_b) {
  // if (size_a == 0) return 0;
  // assert(size_b != 0);
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);        // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;  // warp index within the CTA
  __shared__ T cache[MAX_BLOCK_SIZE];
  cache[warp_lane * WARP_SIZE + thread_lane] =
      b[thread_lane * size_b / WARP_SIZE];
  __syncwarp();
  T num = 0;
  for (auto i = thread_lane; i < size_a; i += WARP_SIZE) {
    auto key = a[i];
    if (!binary_search_2phase(b, cache, key, size_b))
      num += 1;
  }
  return num;
}

template <typename T>
DEV_INLINE T difference_num(T* a, T size_a, T* b, T size_b) {
  return difference_num_bs_cache(a, size_a, b, size_b);
}

// compute set difference: a - b
template <typename T>
DEV_INLINE T difference_num_bs(T* a, T size_a, T* b, T size_b, T upper_bound) {
  // if (size_a == 0) return 0;
  // assert(size_b != 0);
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);  // thread index within the warp
  T num = 0;
  for (auto i = thread_lane; i < size_a; i += WARP_SIZE) {
    auto key = a[i];
    int is_smaller = key < upper_bound ? 1 : 0;
    if (is_smaller && !binary_search(b, key, size_b))
      num += 1;
    unsigned active = __activemask();
    unsigned mask = __ballot_sync(active, is_smaller);
    if (mask != FULL_MASK)
      break;
  }
  return num;
}

template <typename T>
DEV_INLINE T difference_num_bs_cache(T* a, T size_a, T* b, T size_b,
                                     T upper_bound) {
  // if (size_a == 0) return 0;
  // assert(size_b != 0);
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);        // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;  // warp index within the CTA
  __shared__ T cache[MAX_BLOCK_SIZE];
  cache[warp_lane * WARP_SIZE + thread_lane] =
      b[thread_lane * size_b / WARP_SIZE];
  __syncwarp();
  T num = 0;
  for (auto i = thread_lane; i < size_a; i += WARP_SIZE) {
    auto key = a[i];
    int is_smaller = key < upper_bound ? 1 : 0;
    if (is_smaller && !binary_search_2phase(b, cache, key, size_b))
      num += 1;
    unsigned active = __activemask();
    unsigned mask = __ballot_sync(active, is_smaller);
    if (mask != FULL_MASK)
      break;
  }
  return num;
}

template <typename T>
DEV_INLINE T difference_num(T* a, T size_a, T* b, T size_b, T upper_bound) {
  return difference_num_bs_cache(a, size_a, b, size_b, upper_bound);
}

template <typename T>
DEV_INLINE T difference_set_bs(T* a, T size_a, T* b, T size_b, T* c) {
  // if (size_a == 0) return 0;
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);        // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;  // warp index within the CTA
  __shared__ T count[WARPS_PER_BLOCK];

  if (thread_lane == 0)
    count[warp_lane] = 0;
  for (auto i = thread_lane; i < size_a; i += WARP_SIZE) {
    unsigned active = __activemask();
    __syncwarp(active);
    T key = a[i];  // each thread picks a vertex as the key
    int found = 0;
    if (!binary_search(b, key, size_b))
      found = 1;
    unsigned mask = __ballot_sync(active, found);
    auto idx = __popc(mask << (WARP_SIZE - thread_lane - 1));
    if (found)
      c[count[warp_lane] + idx - 1] = key;
    if (thread_lane == 0)
      count[warp_lane] += __popc(mask);
  }
  return count[warp_lane];
}

template <typename T>
DEV_INLINE T difference_set_bs_cache(T* a, T size_a, T* b, T size_b, T* c) {
  // if (size_a == 0) return 0;
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);        // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;  // warp index within the CTA
  __shared__ T count[WARPS_PER_BLOCK];
  __shared__ T cache[MAX_BLOCK_SIZE];
  cache[warp_lane * WARP_SIZE + thread_lane] =
      b[thread_lane * size_b / WARP_SIZE];
  __syncwarp();

  if (thread_lane == 0)
    count[warp_lane] = 0;
  for (auto i = thread_lane; i < size_a; i += WARP_SIZE) {
    unsigned active = __activemask();
    __syncwarp(active);
    T key = a[i];  // each thread picks a vertex as the key
    int found = 0;
    if (!binary_search_2phase(b, cache, key, size_b))
      found = 1;
    unsigned mask = __ballot_sync(active, found);
    auto idx = __popc(mask << (WARP_SIZE - thread_lane - 1));
    if (found)
      c[count[warp_lane] + idx - 1] = key;
    if (thread_lane == 0)
      count[warp_lane] += __popc(mask);
  }
  return count[warp_lane];
}

template <typename T>
DEV_INLINE T difference_set(T* a, T size_a, T* b, T size_b, T* c) {
  return difference_set_bs_cache(a, size_a, b, size_b, c);
}

// set difference: c = a - b
template <typename T>
DEV_INLINE T difference_set_bs(T* a, T size_a, T* b, T size_b, T upper_bound,
                               T* c) {
  // if (size_a == 0) return 0;
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);        // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;  // warp index within the CTA
  __shared__ T count[WARPS_PER_BLOCK];

  if (thread_lane == 0)
    count[warp_lane] = 0;
  for (auto i = thread_lane; i < size_a; i += WARP_SIZE) {
    unsigned active = __activemask();
    __syncwarp(active);
    T key = a[i];  // each thread picks a vertex as the key
    int is_smaller = key < upper_bound ? 1 : 0;
    int found = 0;
    if (is_smaller && !binary_search(b, key, size_b))
      found = 1;
    unsigned mask = __ballot_sync(active, found);
    auto idx = __popc(mask << (WARP_SIZE - thread_lane - 1));
    if (found)
      c[count[warp_lane] + idx - 1] = key;
    if (thread_lane == 0)
      count[warp_lane] += __popc(mask);
    mask = __ballot_sync(active, is_smaller);
    if (mask != FULL_MASK)
      break;
  }
  return count[warp_lane];
}

// set difference: c = a - b
template <typename T>
DEV_INLINE T difference_set_bs_cache(T* a, T size_a, T* b, T size_b,
                                     T upper_bound, T* c) {
  // if (size_a == 0) return 0;
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);        // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;  // warp index within the CTA
  __shared__ T count[WARPS_PER_BLOCK];
  __shared__ T cache[MAX_BLOCK_SIZE];
  cache[warp_lane * WARP_SIZE + thread_lane] =
      b[thread_lane * size_b / WARP_SIZE];
  __syncwarp();
  if (thread_lane == 0)
    count[warp_lane] = 0;
  for (auto i = thread_lane; i < size_a; i += WARP_SIZE) {
    unsigned active = __activemask();
    __syncwarp(active);
    T key = a[i];  // each thread picks a vertex as the key
    int is_smaller = key < upper_bound ? 1 : 0;
    int found = 0;
    if (is_smaller && !binary_search_2phase(b, cache, key, size_b))
      found = 1;
    unsigned mask = __ballot_sync(active, found);
    auto idx = __popc(mask << (WARP_SIZE - thread_lane - 1));
    if (found)
      c[count[warp_lane] + idx - 1] = key;
    if (thread_lane == 0)
      count[warp_lane] += __popc(mask);
    mask = __ballot_sync(active, is_smaller);
    if (mask != FULL_MASK)
      break;
  }
  return count[warp_lane];
}

template <typename T>
DEV_INLINE T difference_set(T* a, T size_a, T* b, T size_b, T upper_bound,
                            T* c) {
  return difference_set_bs_cache(a, size_a, b, size_b, upper_bound, c);
}

template <typename T>
__forceinline__ __device__ T fuse_set_bs_cache(T* a, T size_a, T* b, T size_b,
                                               T* c, T* d, T* e, T* int_map) {
  // if (size_a == 0) return 0;
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);        // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;  // warp index within the CTA
  __shared__ T dif_ab_count[WARPS_PER_BLOCK];
  __shared__ T int_ab_count[WARPS_PER_BLOCK];
  __shared__ T dif_ba_count[WARPS_PER_BLOCK];
  __shared__ T cache[MAX_BLOCK_SIZE];
  auto lookup = a;
  auto search = b;
  auto lookup_size = size_a;
  auto search_size = size_b;
  auto dif_ab = c;
  auto int_ab = d;
  auto dif_ba = e;
  if (size_a > size_b) {
    lookup = b;
    search = a;
    lookup_size = size_b;
    search_size = size_a;
    dif_ab = e;
    dif_ba = c;
  }

  cache[warp_lane * WARP_SIZE + thread_lane] =
      search[thread_lane * search_size / WARP_SIZE];
  __syncwarp();

  for (auto i = thread_lane; i < search_size; i += WARP_SIZE) {
    int_map[i] = 0;
  }
  __syncwarp();

  if (thread_lane == 0) {
    dif_ab_count[warp_lane] = 0;
    int_ab_count[warp_lane] = 0;
    dif_ba_count[warp_lane] = 0;
  }

  for (auto i = thread_lane; i < lookup_size; i += WARP_SIZE) {
    unsigned active = __activemask();
    __syncwarp(active);
    T key = lookup[i];  // each thread picks a vertex as the key
    int found = 0;
    if (!binary_search_2phase_map(search, cache, key, search_size, int_map))
      found = 1;
    unsigned dif_ab_mask = __ballot_sync(active, found);
    auto idx = __popc(dif_ab_mask << (WARP_SIZE - thread_lane - 1));

    unsigned int_ab_mask = __ballot_sync(active, !found);
    auto idx1 = __popc(int_ab_mask << (WARP_SIZE - thread_lane - 1));

    // if (found) c[dif_ab_count[warp_lane]+idx-1] = key;
    // else  d[int_ab_count[warp_lane]+idx1-1] = key;
    if (found)
      dif_ab[dif_ab_count[warp_lane] + idx - 1] = key;
    else
      int_ab[int_ab_count[warp_lane] + idx1 - 1] = key;
    if (thread_lane == 0) {
      dif_ab_count[warp_lane] += __popc(dif_ab_mask);
      int_ab_count[warp_lane] += __popc(int_ab_mask);
    }
  }
  __syncwarp();
  for (auto i = thread_lane; i < search_size; i += WARP_SIZE) {
    unsigned active = __activemask();
    __syncwarp(active);
    int found = !int_map[i];
    unsigned dif_ba_mask = __ballot_sync(active, found);
    auto idx = __popc(dif_ba_mask << (WARP_SIZE - thread_lane - 1));
    // if (found) e[dif_ba_count[warp_lane]+idx-1] = search[i];
    if (found)
      dif_ba[dif_ba_count[warp_lane] + idx - 1] = search[i];
    if (thread_lane == 0) {
      dif_ba_count[warp_lane] += __popc(dif_ba_mask);
    }
  }
  __syncwarp();

  return int_ab_count[warp_lane];
}

template <typename T>
__forceinline__ __device__ T fuse_set(T* a, T size_a, T* b, T size_b, T* c,
                                      T* d, T* e, T* int_map) {
  return fuse_set_bs_cache(a, size_a, b, size_b, c, d, e, int_map);
}

template <typename T>
__forceinline__ __device__ T dif_int_set_bs(T* a, T size_a, T* b, T size_b,
                                            T upper_bound, T* c, T* d) {
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);        // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;  // warp index within the CTA
  __shared__ T dif_ab_count[WARPS_PER_BLOCK];
  __shared__ T int_ab_count[WARPS_PER_BLOCK];

  auto lookup = a;
  auto search = b;
  auto lookup_size = size_a;
  auto search_size = size_b;
  auto dif_ab = c;
  auto int_ab = d;

  if (thread_lane == 0) {
    dif_ab_count[warp_lane] = 0;
    int_ab_count[warp_lane] = 0;
  }

  for (auto i = thread_lane; i < lookup_size; i += WARP_SIZE) {
    unsigned active = __activemask();
    __syncwarp(active);
    T key = lookup[i];  // each thread picks a vertex as the key
    int found_dif = 0;
    int found_int = 0;
    int is_smaller = key < upper_bound ? 1 : 0;
    // if(is_smaller){
    //   if (!binary_search(search, key, search_size))
    //     found_dif = 1;
    //   else
    //     found_int = 1;
    // }

    if (is_smaller && !binary_search(search, key, search_size))
      found_dif = 1;
    else
      found_int = 1;

    unsigned dif_ab_mask = __ballot_sync(active, found_dif);
    auto idx = __popc(dif_ab_mask << (WARP_SIZE - thread_lane - 1));

    unsigned int_ab_mask = __ballot_sync(active, found_int);
    auto idx1 = __popc(int_ab_mask << (WARP_SIZE - thread_lane - 1));

    if (found_dif)
      dif_ab[dif_ab_count[warp_lane] + idx - 1] = key;
    if (found_int)
      int_ab[int_ab_count[warp_lane] + idx1 - 1] = key;
    if (thread_lane == 0) {
      dif_ab_count[warp_lane] += __popc(dif_ab_mask);
      int_ab_count[warp_lane] += __popc(int_ab_mask);
    }
  }

  return dif_ab_count[warp_lane];
}

template <typename T>
__forceinline__ __device__ T dif_int_set(T* a, T size_a, T* b, T size_b,
                                         T upper_bound, T* c, T* d) {
  return dif_int_set_bs(a, size_a, b, size_b, upper_bound, c, d);
}

template <typename T>
__forceinline__ __device__ T dif_int_set_bs(T* a, T size_a, T* b, T size_b,
                                            T* c, T* d) {
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);        // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;  // warp index within the CTA
  __shared__ T dif_ab_count[WARPS_PER_BLOCK];
  __shared__ T int_ab_count[WARPS_PER_BLOCK];

  auto lookup = a;
  auto search = b;
  auto lookup_size = size_a;
  auto search_size = size_b;
  auto dif_ab = c;
  auto int_ab = d;

  if (thread_lane == 0) {
    dif_ab_count[warp_lane] = 0;
    int_ab_count[warp_lane] = 0;
  }

  for (auto i = thread_lane; i < lookup_size; i += WARP_SIZE) {
    unsigned active = __activemask();
    __syncwarp(active);
    T key = lookup[i];  // each thread picks a vertex as the key
    int found_dif = 0;
    int found_int = 0;

    if (!binary_search(search, key, search_size))
      found_dif = 1;
    else
      found_int = 1;

    unsigned dif_ab_mask = __ballot_sync(active, found_dif);
    auto idx = __popc(dif_ab_mask << (WARP_SIZE - thread_lane - 1));

    unsigned int_ab_mask = __ballot_sync(active, found_int);
    auto idx1 = __popc(int_ab_mask << (WARP_SIZE - thread_lane - 1));

    if (found_dif)
      dif_ab[dif_ab_count[warp_lane] + idx - 1] = key;
    if (found_int)
      int_ab[int_ab_count[warp_lane] + idx1 - 1] = key;
    if (thread_lane == 0) {
      dif_ab_count[warp_lane] += __popc(dif_ab_mask);
      int_ab_count[warp_lane] += __popc(int_ab_mask);
    }
  }

  return dif_ab_count[warp_lane];
}

template <typename T>
__forceinline__ __device__ T dif_int_set(T* a, T size_a, T* b, T size_b, T* c,
                                         T* d) {
  return dif_int_set_bs(a, size_a, b, size_b, c, d);
}

#endif