#define INTERLEAVE_CATCH(idx, stride) idx += stride;

#define WARP_DYNAMIC_CATCH(idx, stride) \
  if (thread_lane == 0)                 \
    idx = atomicAdd(stride, 1);         \
  __syncwarp();                         \
  idx = __shfl_sync(0xffffffff, idx, 0);

#define BLOCK_DYNAMIC_CATCH(shared_idx, local_idx, stride) \
  if (threadIdx.x == 0)                                    \
    shared_idx = atomicAdd(stride, 1);                     \
  __syncthreads();                                         \
  local_idx = shared_idx;

#define DYNAMIC_MODE
#ifdef DYNAMIC_MODE
#define WARP_NEXT_WORK_CATCH(idx, stride1, stride2) \
  WARP_DYNAMIC_CATCH(idx, stride1);
#define BLOCK_NEXT_WORK_CATCH(shared_idx, local_idx, stride1, stride2) \
  BLOCK_DYNAMIC_CATCH(shared_idx, local_idx, stride1);
#define NEXT_WORK_CATCH(idx, stride1, stride2) \
  WARP_NEXT_WORK_CATCH(idx, stride1, stride2)
#else
#define WARP_NEXT_WORK_CATCH(idx, stride1, stride2) \
  INTERLEAVE_CATCH(idx, stride2);
#define BLOCK_NEXT_WORK_CATCH(shared_idx, local_idx, stride1, stride2) \
  INTERLEAVE_CATCH(local_idx, stride2);
#define NEXT_WORK_CATCH(idx, stride1, stride2) \
  WARP_NEXT_WORK_CATCH(idx, stride1, stride2)
#endif

// Special Schedules for Tuning Motif Counting
#define NORMAL_INCREMENT_CHECK_VLIST_DISCONNECT()                          \
  for (VID i = 0; i < list_size[warp_lane]; i++) {                         \
    VID v2 = vlist[i];                                                     \
    VID v2_size = g.getOutDegree(v2);                                      \
    for (auto i = thread_lane; i < list_size[warp_lane]; i += WARP_SIZE) { \
      auto key = vlist[i];                                                 \
      int is_smaller = key < v2 ? 1 : 0;                                   \
      if (is_smaller && !binary_search(g.getNeighbor(v2), key, v2_size))   \
        count += 1;                                                        \
      unsigned active = __activemask();                                    \
      unsigned mask = __ballot_sync(active, is_smaller);                   \
      if (mask != FULL_MASK)                                               \
        break;                                                             \
    }                                                                      \
  }

#define NORMAL_DECREMENT_CHECK_VLIST_DISCONNECT()                              \
  VID other_count = 0;                                                         \
  for (VID i = 0; i < other_size[warp_lane]; i++) {                            \
    VID v2 = vlist[max_deg + i];                                               \
    VID v2_size = g.getOutDegree(v2);                                          \
    for (auto j = thread_lane; j < i + list_size[warp_lane]; j += WARP_SIZE) { \
      int work_id = j;                                                         \
      auto addr = &vlist[max_deg];                                             \
      if (work_id >= i) {                                                      \
        work_id -= i;                                                          \
        addr = vlist;                                                          \
      }                                                                        \
      auto key = addr[work_id];                                                \
      if (!binary_search(g.getNeighbor(v2), key, v2_size)) {                   \
        other_count += 1;                                                      \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  __syncwarp();                                                                \
  auto n = warp_reduce<AccType>(other_count);                                  \
  if (thread_lane == 0) {                                                      \
    VID total_count = count_list[v0];                                          \
    VID true_count = total_count - n;                                          \
    reverse_count += true_count;                                               \
  }

#define FUSE_LOOP_INCREMENT_CHECK_VLIST_DISCONNECT()                   \
  VID size_a = list_size[warp_lane];                                   \
  VID loop_size_a = size_a * size_a;                                   \
  for (VID i = thread_lane; i < loop_size_a; i += WARP_SIZE) {         \
    VID key = vlist[i % size_a];                                       \
    VID v2 = vlist[i / size_a];                                        \
    VID v2_size = g.getOutDegree(v2);                                  \
    int is_smaller = key < v2 ? 1 : 0;                                 \
    if (is_smaller && !binary_search(g.getNeighbor(v2), key, v2_size)) \
      count += 1;                                                      \
  }

#define vlist_threshold 50
#define vlist_percent 0.6
#define USE_NORMAL_CHECK_VLIST(vlist_size) (vlist_size) > (vlist_threshold)

#define USE_INCREMENT_MODE(vlist_size, max_size) \
  ((vlist_size) < ((max_size) *vlist_percent))

// Special Schedules for Tuning Triangle Counting
#define HASH_LOOKUP(thd_start, stride)                                        \
  {                                                                           \
    for (auto e = v_start; e < v_end; e++) {                                  \
      auto u = g.get_edge_dst(e);                                             \
      int u_start = g.edge_begin(u);                                          \
      int u_end = g.edge_end(u);                                              \
      for (auto i = (thd_start) + u_start; i < u_end; i += (stride)) {        \
        int w = g.get_edge_dst(i);                                            \
        int key = w & (bucket_module);                                        \
        P_counter += linear_search_sm(w, shared_partition, partition,         \
                                      bin_count, key + binOffset, BIN_START); \
      }                                                                       \
    }                                                                         \
  }

#define WHOLE_BLOCK_HASH_LOOKUP() HASH_LOOKUP(threadIdx.x, blockDim.x)

#define SUB_BLOCK_HASH_LOOKUP()                                               \
  {                                                                           \
    now = g.edge_begin(vertex);                                               \
    int superwarp_ID = threadIdx.x / 64;                                      \
    int superwarp_TID = threadIdx.x % 64;                                     \
    int workid = superwarp_TID;                                               \
    now = now + superwarp_ID;                                                 \
    int neighbor = g.get_edge_dst(now);                                       \
    int neighbor_start = g.edge_begin(neighbor);                              \
    int neighbor_degree = g.edge_end(neighbor) - neighbor_start;              \
    while (now < v_end) {                                                     \
      while (now < v_end && workid >= neighbor_degree) {                      \
        now += 16;                                                            \
        workid -= neighbor_degree;                                            \
        neighbor = g.get_edge_dst(now);                                       \
        neighbor_start = g.edge_begin(neighbor);                              \
        neighbor_degree = g.edge_end(neighbor) - neighbor_start;              \
      }                                                                       \
      if (now < v_end) {                                                      \
        int temp = g.get_edge_dst(neighbor_start + workid);                   \
        int bin = temp & bucket_module;                                       \
        P_counter += linear_search_sm(temp, shared_partition, partition,      \
                                      bin_count, bin + binOffset, BIN_START); \
      }                                                                       \
      workid += 64;                                                           \
    }                                                                         \
  }

#define WHOLE_WARP_HASH_LOOKUP() HASH_LOOKUP(thread_lane, WARP_SIZE)
#define SUB_WARP_HASH_LOOKUP()                                                \
  {                                                                           \
    int e = v_start;                                                          \
    int workid = thread_lane;                                                 \
    while (e < v_end) {                                                       \
      int u = g.get_edge_dst(e);                                              \
      int u_start = g.edge_begin(u);                                          \
      int u_degree = g.edge_end(u) - u_start;                                 \
      while (e < v_end && workid >= u_degree) {                               \
        e++;                                                                  \
        workid -= u_degree;                                                   \
        u = g.get_edge_dst(e);                                                \
        u_start = g.edge_begin(u);                                            \
        u_degree = g.edge_end(u) - u_start;                                   \
      }                                                                       \
      if (e < v_end) {                                                        \
        int w = g.get_edge_dst(u_start + workid);                             \
        int key = w & bucket_module;                                          \
        P_counter += linear_search_sm(w, shared_partition, partition,         \
                                      bin_count, key + binOffset, BIN_START); \
      }                                                                       \
      __syncwarp();                                                           \
      e = __shfl_sync(0xffffffff, e, 31);                                     \
      workid = __shfl_sync(0xffffffff, workid, 31);                           \
      workid += thread_lane + 1;                                              \
    }                                                                         \
  }

#define INTRA_BLOCK_PROCESS
#ifdef INTRA_BLOCK_PROCESS
#define BLOCK_HASH_LOOKUP() SUB_BLOCK_HASH_LOOKUP()
#else
#define BLOCK_HASH_LOOKUP() WHOLE_BLOCK_HASH_LOOKUP()
#endif

#define INTRA_WARP_PROCESS
#ifdef INTRA_WARP_PROCESS
#define WARP_HASH_LOOKUP() SUB_WARP_HASH_LOOKUP()
#else
#define WARP_HASH_LOOKUP() WHOLE_WARP_HASH_LOOKUP()
#endif

#define HASH_INSERT(idx_start, stride)                                       \
  for (auto e = idx_start + v_start; e < v_end; e += stride) {               \
    int u = g.get_edge_dst(e);                                               \
    int key = u & bucket_module;                                             \
    int index = atomicAdd(&bin_count[key + binOffset], 1);                   \
    if (index < SHARED_BUCKET_SIZE) {                                        \
      shared_partition[index * BLOCK_BUCKET_NUM + key + binOffset] = u;      \
    } else if (index < TR_BUCKET_SIZE) {                                     \
      index = index - SHARED_BUCKET_SIZE;                                    \
      partition[index * BLOCK_BUCKET_NUM + binOffset + key + BIN_START] = u; \
    }                                                                        \
  }
#define SHARED_BUCKET_SIZE 6
#define USE_CTA 100
#define USE_WARP 2

#define BLOCK_BUCKET_NUM 1024
#define WARP_BUCKET_NUM 32

#define TR_BUCKET_SIZE 100
