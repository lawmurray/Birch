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

#include <iostream>

/*
 * Disable retention of extents by jemalloc. This is critical as the custom
 * extent hooks for the CUDA backend allocate physical memory---which should
 * not be retained---rather than virtual memory.
 */
const char* numbirch_malloc_conf = "retain:false";

namespace numbirch {

/**
 * @internal
 * 
 * Event. Pointers to objects of this type are type-erased to `void*` for use
 * by the various event management functions.
 */
struct Event {
  /**
   * Event.
   */
  cudaEvent_t event;

  /**
   * Stream on which the event is recorded.
   */
  cudaStream_t stream;

  Event() :
      event(0),
      stream(0) {
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  }

  ~Event() {
    CUDA_CHECK(cudaEventDestroy(event));
  }
};

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

void* event_extent_alloc(extent_hooks_t *extent_hooks, void *new_addr,
    size_t size, size_t alignment, bool *zero, bool *commit,
    unsigned arena_ind) {
  assert(!*zero);
  if (!new_addr) {
    CUDA_CHECK(cudaMallocHost(&new_addr, size));
    *commit = true;
  }
  Event* evts = static_cast<Event*>(new_addr);
  for (size_t i = 0; i < size/sizeof(Event); ++i) {
    new (evts + i) Event();
  }
  return new_addr;
}

bool event_extent_dalloc(extent_hooks_t *extent_hooks, void *addr,
    size_t size, bool committed, unsigned arena_ind) {
  Event* evts = static_cast<Event*>(addr);
  for (size_t i = 0; i < size/sizeof(Event); ++i) {
    evts[i].~Event();
  }
  CUDA_CHECK(cudaFreeHost(addr));
  return false;
}

void event_extent_destroy(extent_hooks_t *extent_hooks, void *addr,
    size_t size, bool committed, unsigned arena_ind) {
  Event* evts = static_cast<Event*>(addr);
  for (size_t i = 0; i < size/sizeof(Event); ++i) {
    evts[i].~Event();
  }
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

void* event_create() {
  return event_malloc(sizeof(Event));
}

void event_destroy(void* evt) {
  return event_free(evt, sizeof(Event));
}

void event_record_read(void* evt) {
  assert(evt != 0);
  Event* e = static_cast<Event*>(evt);
  if (e->stream == stream || e->stream == 0) {
    /* just update the record in this stream */
    CUDA_CHECK(cudaEventRecord(e->event, stream));
    e->stream = stream;
  } else {
    /* concurrent reads are permitted; evt should be such that waiting or
     * joining on it ensures that *all* reads are finished; this is
     * accomplished with the auxiliary streams; first, join this thread's
     * auxiliary stream with the existing read event (which may be on a
     * different stream), blocking the auxiliary stream on all previous reads
     */
    if (e->stream != aux_stream) {
      CUDA_CHECK(cudaStreamWaitEvent(aux_stream, e->event));
    }

    /* record a new event to indicate when the new read finishes */
    CUDA_CHECK(cudaEventRecord(e->event, stream));
    
    /* join the auxiliary stream to that new event, so that it is now blocked
     * on all previous reads and the new read */
    CUDA_CHECK(cudaStreamWaitEvent(aux_stream, e->event));

    /* record a new event in the auxiliary stream to indicate when all
     * previous reads and the new read are complete */
    CUDA_CHECK(cudaEventRecord(e->event, aux_stream));
    e->stream = aux_stream;
  }
}

void event_record_write(void* evt) {
  assert(evt != 0);
  Event* e = static_cast<Event*>(evt);
  CUDA_CHECK(cudaEventRecord(e->event, stream));
  e->stream = stream;
}

bool event_test(void* evt) {
  assert(evt != 0);
  Event* e = static_cast<Event*>(evt);
  cudaError_t err = cudaEventQuery(e->event);
  return err == cudaSuccess;
}

void event_wait(void* evt) {
  assert(evt != 0);
  Event* e = static_cast<Event*>(evt);
  CUDA_CHECK(cudaEventSynchronize(e->event));
}

void event_join(void* evt) {
  assert(evt != 0);
  Event* e = static_cast<Event*>(evt);

  /* if the event is already in the current stream then no need to join as
   * operations will follow it anyway; avoid the CUDA API call */
  if (e->stream != stream) {
    CUDA_CHECK(cudaStreamWaitEvent(stream, e->event));
  }
}

}
