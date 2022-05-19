/**
 * @file
 */
#include "numbirch/jemalloc/jemalloc.hpp"

#include <cassert>
#include <cstring>
#include <omp.h>

/*
 * Unified shared arena and thread cache for jemalloc. Allocations in this
 * arena will migrate between host and device on demand.
 */
static thread_local unsigned shared_arena = 0;
static thread_local unsigned shared_tcache = 0;
static thread_local int shared_flags = 0;

/*
 * Device arena and thread cache for jemalloc. These use the same custom
 * allocation hooks as for shared allocations (i.e. unified shared memory),
 * and so can be accessed from the host, but by using a separate arena, we
 * expect these allocations to gravitate onto the device and stay there, even
 * when reused.
 */
static thread_local unsigned device_arena = 0;
static thread_local unsigned device_tcache = 0;
static thread_local int device_flags = 0;

/*
 * Host arena and thread cache for jemalloc. These use host memory only.
 */
static thread_local unsigned host_arena = 0;
static thread_local unsigned host_tcache = 0;
static thread_local int host_flags = 0;

/**
 * Custom extent hooks structure.
 */
static extent_hooks_t hooks = {
  numbirch::extent_alloc,
  nullptr,
  numbirch::extent_destroy,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr
};

/**
 * Custom extent hooks structure.
 */
static extent_hooks_t device_hooks = {
  numbirch::device_extent_alloc,
  nullptr,
  numbirch::device_extent_destroy,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr
};

/**
 * Custom extent hooks structure.
 */
static extent_hooks_t host_hooks = {
  numbirch::host_extent_alloc,
  nullptr,
  numbirch::host_extent_destroy,
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

void numbirch::jemalloc_init() {
  #pragma omp parallel num_threads(omp_get_max_threads())
  {
    /* shared arena setup */
    shared_arena = make_arena(&hooks);
    shared_tcache = make_tcache();
    shared_flags = MALLOCX_ARENA(shared_arena)|MALLOCX_TCACHE(shared_tcache);

    /* device arena setup */
    device_arena = make_arena(&device_hooks);
    device_tcache = make_tcache();
    device_flags = MALLOCX_ARENA(device_arena)|MALLOCX_TCACHE(device_tcache);

    /* host arena setup */
    host_arena = make_arena(&host_hooks);
    host_tcache = make_tcache();
    host_flags = MALLOCX_ARENA(host_arena)|MALLOCX_TCACHE(host_tcache);
  }
}

void numbirch::jemalloc_term() {
  ///@todo
}

void* numbirch::device_malloc(const size_t size) {
  assert(device_arena > 0);
  return size == 0 ? nullptr : numbirch_mallocx(size, device_flags);
}

void numbirch::device_free(void* ptr) {
  assert(device_arena > 0);
  if (ptr) {
    numbirch_dallocx(ptr, device_flags);
  }
}

void* numbirch::host_malloc(const size_t size) {
  assert(host_arena > 0);
  return size == 0 ? nullptr : numbirch_mallocx(size, host_flags);
}

void numbirch::host_free(void* ptr) {
  assert(host_arena > 0);
  if (ptr) {
    numbirch_dallocx(ptr, host_flags);
  }
}
