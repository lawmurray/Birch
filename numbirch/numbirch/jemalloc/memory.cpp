/**
 * @file
 */
#include "numbirch/jemalloc/jemalloc.hpp"
#include "numbirch/memory.hpp"

#include <cassert>
#include <cstring>

#if HAVE_OMP_H
#include <omp.h>
#endif

namespace numbirch {
/*
 * Unified shared arena and thread cache. Allocations in this arena will
 * migrate between host and device on demand.
 */
static thread_local unsigned shared_arena = 0;
static thread_local unsigned shared_tcache = 0;
static thread_local int shared_flags = 0;

/*
 * Device arena and thread cache. These use the same custom allocation hooks
 * as for shared allocations (i.e. unified shared memory), and so can be
 * accessed from the host, but by using a separate arena, we expect these
 * allocations to gravitate onto the device and stay there, even when reused.
 */
static thread_local unsigned device_arena = 0;
static thread_local unsigned device_tcache = 0;
static thread_local int device_flags = 0;

/*
 * Host arena and thread cache. These use host memory only.
 */
static thread_local unsigned host_arena = 0;
static thread_local unsigned host_tcache = 0;
static thread_local int host_flags = 0;

/*
 * MIB for "arenas.lookup" to avoid repeated name lookups.
 */
static size_t mib[2] = {0, 0};

/**
 * Custom extent hooks for shared arena.
 */
static extent_hooks_t hooks = {
  extent_alloc,
  nullptr,
  extent_destroy,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr
};

/**
 * Custom extent hooks for device arena.
 */
static extent_hooks_t device_hooks = {
  device_extent_alloc,
  nullptr,
  device_extent_destroy,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr
};

/**
 * Custom extent hooks for host arena.
 */
static extent_hooks_t host_hooks = {
  host_extent_alloc,
  nullptr,
  host_extent_destroy,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr
};

static unsigned make_arena(extent_hooks_t* hooks) {
  [[maybe_unused]] int ret;
  unsigned arena = 0;
  size_t size = sizeof(arena);
  ret = numbirch_mallctl("arenas.create", &arena, &size, &hooks,
      sizeof(hooks));
  assert(ret == 0);
  return arena;
}

static unsigned make_tcache() {
  [[maybe_unused]] int ret;
  unsigned tcache = 0;
  size_t size = sizeof(tcache);
  ret = numbirch_mallctl("tcache.create", &tcache, &size, nullptr, 0);
  assert(ret == 0);
  return tcache;
}

void jemalloc_init() {
  /* convert "arenas.lookup" name to mib for faster lookups in future */
  [[maybe_unused]] int ret;
  size_t miblen = 2;
  ret = numbirch_mallctlnametomib("arenas.lookup", mib, &miblen);
  assert(ret == 0);

  #pragma omp parallel num_threads(omp_get_max_threads())
  {
    /* shared arena and thread cache setup */
    shared_arena = make_arena(&hooks);
    shared_tcache = make_tcache();
    shared_flags = MALLOCX_ARENA(shared_arena)|MALLOCX_TCACHE(shared_tcache);

    /* device arena and thread cache setup */
    device_arena = make_arena(&device_hooks);
    device_tcache = make_tcache();
    device_flags = MALLOCX_ARENA(device_arena)|MALLOCX_TCACHE(device_tcache);

    /* host arena and thread cache setup */
    host_arena = make_arena(&host_hooks);
    host_tcache = make_tcache();
    host_flags = MALLOCX_ARENA(host_arena)|MALLOCX_TCACHE(host_tcache);
  }
}

void jemalloc_term() {
  ///@todo
}

void* shared_malloc(const size_t size) {
  assert(shared_arena > 0);
  return size == 0 ? nullptr : numbirch_mallocx(size, shared_flags);
}

void shared_free(void* ptr) {
  assert(shared_arena > 0);
  if (ptr) {
    numbirch_dallocx(ptr, shared_flags);
  }
}

void shared_free(void* ptr, const size_t size) {
  assert(shared_arena > 0);
  if (ptr) {
    numbirch_sdallocx(ptr, size, shared_flags);
  }
}

void shared_free_async(void* ptr) {
  assert(shared_arena == 0 && shared_flags == 0);
  // ^ should be called by an external thread (e.g. of the CUDA runtime)
  numbirch_dallocx(ptr, MALLOCX_TCACHE_NONE);
  // ^ skip the thread cache, as an external thread will never malloc()
}


void* device_malloc(const size_t size) {
  assert(device_arena > 0);
  return size == 0 ? nullptr : numbirch_mallocx(size, device_flags);
}

void device_free(void* ptr) {
  assert(device_arena > 0);
  if (ptr) {
    numbirch_dallocx(ptr, device_flags);
  }
}

void device_free(void* ptr, const size_t size) {
  assert(device_arena > 0);
  if (ptr) {
    numbirch_sdallocx(ptr, size, device_flags);
  }
}

void* host_malloc(const size_t size) {
  assert(host_arena > 0);
  return size == 0 ? nullptr : numbirch_mallocx(size, host_flags);
}

void host_free(void* ptr) {
  assert(host_arena > 0);
  if (ptr) {
    numbirch_dallocx(ptr, host_flags);
  }
}

void host_free(void* ptr, const size_t size) {
  assert(host_arena > 0);
  if (ptr) {
    numbirch_sdallocx(ptr, size, host_flags);
  }
}

void* realloc(void* oldptr, const size_t oldsize, const size_t newsize) {
  void* newptr = nullptr;
  if (newsize == 0) {
    free(oldptr);
  } else if (newsize == oldsize) {
    newptr = oldptr;
  } else if (newsize < oldsize) {
    /* shrinking, should be able to maintain current allocation */
    newptr = numbirch_rallocx(oldptr, newsize, shared_flags);
    assert(newptr == oldptr);
  } else {
    /* growing, maintain current allocation if possible */
    size_t extra = newsize - oldsize;
    size_t size = numbirch_xallocx(oldptr, oldsize, extra, shared_flags);
    if (size == newsize) {
      /* successful */
      newptr = oldptr;
    } else {
      /* cannot extend in place, allocate a new buffer and copy with device
       * functions, rather than using rallocx(), which will copy with host
       * functions internally */
      newptr = shared_malloc(newsize);
      memcpy(newptr, oldptr, oldsize);
      free(oldptr, oldsize);
    }
  }
  return newptr;
}

}
