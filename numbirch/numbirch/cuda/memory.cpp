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
#include "numbirch/array/ArrayControl.hpp"

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
    shared_free(ptr);
  }
}

void free(void* ptr, const size_t size) {
  if (ptr) {
    shared_free(ptr, size);
  }
}

void memcpy(void* dst, const void* src, size_t n) {
  CUDA_CHECK(cudaMemcpyAsync(dst, src, n, cudaMemcpyDefault, stream));
}

void array_init(ArrayControl* ctl, const size_t bytes) {
  assert(ctl);
  ctl->buf = numbirch::malloc(bytes);
  ctl->bytes = bytes;
  ctl->streamAlloc = stream;
  ctl->streamWrite = stream;
}

void array_term(ArrayControl* ctl) {
  assert(ctl);
  if (ctl->buf) {
    auto streamAlloc = static_cast<cudaStream_t>(ctl->streamAlloc);
    auto streamWrite = static_cast<cudaStream_t>(ctl->streamWrite);
    if (streamWrite != streamAlloc) {
      /* ensure that the most recent write completes before reuse */
      CUDA_CHECK(cudaEventRecord(event, streamWrite));
      CUDA_CHECK(cudaStreamWaitEvent(streamAlloc, event));
    }
    numbirch::free(ctl->buf, ctl->bytes);
  }
}

void array_resize(ArrayControl* ctl, const size_t bytes) {
  ctl->buf = numbirch::realloc(ctl->buf, ctl->bytes, bytes);
  ctl->bytes = bytes;
}

void array_copy(ArrayControl* dst, const ArrayControl* src) {
  auto src1 = const_cast<ArrayControl*>(src);
  before_read(src1);
  before_write(dst);
  memcpy(dst->buf, src1->buf, std::min(dst->bytes, src1->bytes));
  after_write(dst);
  after_read(src1);
}

void array_wait(ArrayControl* ctl) {
  assert(ctl);
  auto streamWrite = static_cast<cudaStream_t>(ctl->streamWrite);
  CUDA_CHECK(cudaStreamSynchronize(streamWrite));
}

bool array_test(ArrayControl* ctl) {
  assert(ctl);
  auto streamWrite = static_cast<cudaStream_t>(ctl->streamWrite);
  return cudaStreamQuery(streamWrite) == cudaSuccess;
}

void before_read(ArrayControl* ctl) {
  assert(ctl);
  auto streamWrite = static_cast<cudaStream_t>(ctl->streamWrite);
  if (streamWrite != stream) {
    /* ensure that the most recent write completes before reading */
    CUDA_CHECK(cudaEventRecord(event, streamWrite));
    CUDA_CHECK(cudaStreamWaitEvent(stream, event));
  }
}

void before_write(ArrayControl* ctl) {
  assert(ctl);
  auto streamWrite = static_cast<cudaStream_t>(ctl->streamWrite);
  if (streamWrite != stream) {
    /* ensure that the most recent write completes before writing */
    CUDA_CHECK(cudaEventRecord(event, streamWrite));
    CUDA_CHECK(cudaStreamWaitEvent(stream, event));
  }
}

void after_read(ArrayControl* ctl) {
  assert(ctl);
  ctl->streamWrite = stream;
}

void after_write(ArrayControl* ctl) {
  assert(ctl);
  ctl->streamWrite = stream;
}

}
