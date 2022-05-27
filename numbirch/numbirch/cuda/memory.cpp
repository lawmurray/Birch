/**
 * @file
 */
#include "numbirch/memory.hpp"
#include "numbirch/random.hpp"
#include "numbirch/cuda/cuda.hpp"
#include "numbirch/cuda/cublas.hpp"
#include "numbirch/cuda/cusolver.hpp"
#include "numbirch/cuda/curand.hpp"
#include "numbirch/jemalloc/jemalloc.hpp"

/*
 * Disable retention of extents by jemalloc. This is critical as the custom
 * extent hooks for the CUDA backend allocate physical memory---which should
 * not be retained---rather than virtual memory.
 */
const char* numbirch_malloc_conf = "retain:false";

namespace numbirch {

void init() {
  cuda_init();
  cublas_init();
  cusolver_init();
  curand_init();
  jemalloc_init();
  seed();
}

void wait() {
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

void term() {
  jemalloc_term();
  curand_term();
  cusolver_term();
  cublas_term();
  cuda_term();
}

void* record() {
  cudaEvent_t evt;
  CUDA_CHECK(cudaEventCreate(&evt));
  CUDA_CHECK(cudaEventRecord(evt, stream));
  return (void*)evt;
}

void wait(void* evt) {
  if (evt != 0) {
    cudaEvent_t e = static_cast<cudaEvent_t>(evt);
    CUDA_CHECK(cudaEventSynchronize(e));
  }
}

void forget(void* evt) {
  if (evt != 0) {
    cudaEvent_t e = static_cast<cudaEvent_t>(evt);
    CUDA_CHECK(cudaEventDestroy(e));
  }
}

void* extent_alloc(extent_hooks_t *extent_hooks, void *new_addr, size_t size,
    size_t alignment, bool *zero, bool *commit, unsigned arena_ind) {
  if (!new_addr) {
    CUDA_CHECK(cudaMallocManaged(&new_addr, size));
    *commit = true;
  }
  if (*zero) {
    CUDA_CHECK(cudaMemset(new_addr, 0, size));
  }
  return new_addr;
}

bool extent_dalloc(extent_hooks_t *extent_hooks, void *addr, size_t size,
    bool committed, unsigned arena_ind) {
  CUDA_CHECK(cudaFree(addr));
  return false;
}

void extent_destroy(extent_hooks_t *extent_hooks, void *addr, size_t size,
    bool committed, unsigned arena_ind) {
  CUDA_CHECK(cudaFree(addr));
}

void* device_extent_alloc(extent_hooks_t *extent_hooks, void *new_addr,
    size_t size, size_t alignment, bool *zero, bool *commit,
    unsigned arena_ind) {
  if (!new_addr) {
    CUDA_CHECK(cudaMallocManaged(&new_addr, size));
    *commit = true;
  }
  if (*zero) {
    CUDA_CHECK(cudaMemset(new_addr, 0, size));
  }
  return new_addr;
}

bool device_extent_dalloc(extent_hooks_t *extent_hooks, void *addr,
    size_t size, bool committed, unsigned arena_ind) {
  CUDA_CHECK(cudaFree(addr));
  return false;
}

void device_extent_destroy(extent_hooks_t *extent_hooks, void *addr,
    size_t size, bool committed, unsigned arena_ind) {
  CUDA_CHECK(cudaFree(addr));
}

void* host_extent_alloc(extent_hooks_t *extent_hooks, void *new_addr,
    size_t size, size_t alignment, bool *zero, bool *commit,
    unsigned arena_ind) {
  if (!new_addr) {
    CUDA_CHECK(cudaMallocHost(&new_addr, size));
    *commit = true;
  }
  if (*zero) {
    CUDA_CHECK(cudaMemset(new_addr, 0, size));
  }
  return new_addr;
}

bool host_extent_dalloc(extent_hooks_t *extent_hooks, void *addr,
    size_t size, bool committed, unsigned arena_ind) {
  CUDA_CHECK(cudaFreeHost(addr));
  return false;
}

void host_extent_destroy(extent_hooks_t *extent_hooks, void *addr,
    size_t size, bool committed, unsigned arena_ind) {
  CUDA_CHECK(cudaFreeHost(addr));
}

void* malloc(const size_t size) {
  return shared_malloc(size);
}

void free(void* ptr) {
  if (ptr) {
    if (shared_owns(ptr)) {
      /* as the allocation is from the shared arena of this thread, it can be
       * returned to the pool immediately and safely returned by malloc()
       * again, even before preceding asynchronous operations that use it have
       * completed; threads provide consistent ordering internally, and this
       * allows allocations and deallocations of the same block to be streamed
       * and so minimize overall memory use */
      shared_free(ptr);
    } else {
      /* as the allocation is from the shared arena of a different thread, its
       * return to the pool should be inserted into the stream, to ensure that
       * all asynchronous operations associated with this thread have finished
       * with it before it is return by malloc() again on that other thread;
       * this ensures consistent ordering across threads */
      CUDA_CHECK(cudaLaunchHostFunc(stream, &shared_free_async, ptr));
    }
  }
}

void free(void* ptr, const size_t size) {
  /* see free() above for comments, this version merely inserts size */
  if (ptr) {
    if (shared_owns(ptr)) {
      shared_free(ptr, size);
    } else {
      CUDA_CHECK(cudaLaunchHostFunc(stream, &shared_free_async, ptr));
    }
  }
}

void memcpy(void* dst, const void* src, size_t n) {
  CUDA_CHECK(cudaMemcpyAsync(dst, src, n, cudaMemcpyDefault, stream));
}

}
