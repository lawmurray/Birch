/**
 * @file
 * 
 * oneAPI implementation of non-generic functions.
 */
#include "numbirch/memory.hpp"
#include "numbirch/oneapi/sycl.hpp"
#include "numbirch/jemalloc/jemalloc.hpp"

#if HAVE_OMP_H
#include <omp.h>
#endif

namespace numbirch {

void* extent_alloc(extent_hooks_t *extent_hooks, void *new_addr, size_t size,
    size_t alignment, bool *zero, bool *commit, unsigned arena_ind) {
  if (!new_addr) {
    new_addr = sycl::malloc_shared(size, queue);
  }
  if (*zero) {
    queue.set(new_addr, 0, size);
  }
  queue.wait();
  return new_addr;
}

bool extent_dalloc(extent_hooks_t *extent_hooks, void *addr, size_t size,
    bool committed, unsigned arena_ind) {
  sycl::free(addr, queue);
  return false;
}

void extent_destroy(extent_hooks_t *extent_hooks, void *addr, size_t size,
    bool committed, unsigned arena_ind) {
  sycl::free(addr, queue);
}

void* device_extent_alloc(extent_hooks_t *extent_hooks, void *new_addr,
    size_t size, size_t alignment, bool *zero, bool *commit,
    unsigned arena_ind) {
  if (!new_addr) {
    new_addr = sycl::malloc_shared(size, queue);
  }
  if (*zero) {
    queue.set(new_addr, 0, size);
  }
  queue.wait();
  return new_addr;
}

bool device_extent_dalloc(extent_hooks_t *extent_hooks, void *addr,
    size_t size, bool committed, unsigned arena_ind) {
  sycl::free(addr, queue);
  return false;
}

void device_extent_destroy(extent_hooks_t *extent_hooks, void *addr,
    size_t size, bool committed, unsigned arena_ind) {
  sycl::free(addr, queue);
}

void init() {
  #pragma omp parallel num_threads(omp_get_max_threads())
  {
    device = sycl::device(sycl::gpu_selector());
    context = sycl::context(device);
    queue = sycl::queue(context, device, sycl::property::queue::in_order());
  }
  jemalloc_init();
}

void term() {
  jemalloc_term();
}

void memcpy(void* dst, const size_t dpitch, const void* src,
    const size_t spitch, const size_t width, const size_t height) {
  if (dpitch == width && spitch == width) {
    queue.memcpy(dst, src, width*height);
  } else for (int i = 0; i < height; ++i) {
    queue.memcpy((char*)dst + i*dpitch, (char*)src + i*spitch, width);
  }
}

void wait() {
  queue.wait();
}

}
