/**
 * @file
 * 
 * Jemalloc integration. Implements malloc(), realloc() and free(). Backends
 * using jemalloc should implement extent_alloc(), extent_dalloc() and
 * extent_destroy() instead, as well as call jemalloc_init() during init() and
 * jemalloc_term() during term().
 */
#pragma once

#include <jemalloc/jemalloc_numbirch.h>

namespace numbirch {
/*
 * Initialize jemalloc integrations. This should be called during init() by
 * the backend.
 */
void jemalloc_init();

/*
 * Terminate jemalloc integrations. This should be called during term() by the
 * backend.
 */
void jemalloc_term();

/*
 * Allocate for the shared arena.
 */
void* shared_malloc(const size_t size);

/*
 * Free for the shared arena.
 */
void shared_free(void* ptr);

/*
 * Free for the shared arena.
 */
void shared_free(void* ptr, const size_t size);

/*
 * Free for the shared arena, for use by external threads only. One use case
 * is to insert calls of this into CUDA streams with cudaLaunchHostFunc(). A
 * thread of the CUDA runtime will then call it to execute the free, but that
 * thread will not have its own jemalloc arena or thread cache to use.
 */
void shared_free_async(void* ptr);

/*
 * Allocate for the device arena.
 */
void* device_malloc(const size_t size);

/*
 * Free for the device arena.
 */
void device_free(void* ptr);

/*
 * Free for the device arena.
 */
void device_free(void* ptr, const size_t size);

/*
 * Allocate for the host arena.
 */
void* host_malloc(const size_t size);

/*
 * Free for the host arena.
 */
void host_free(void* ptr);

/*
 * Free for the host arena.
 */
void host_free(void* ptr, const size_t size);

/*
 * Custom alloc() extent hook. This is implemented by the specific backend.
 */
void* extent_alloc(extent_hooks_t *extent_hooks, void *new_addr, size_t size,
    size_t alignment, bool *zero, bool *commit, unsigned arena_ind);

/*
 * Custom dalloc() extent hook. This is implemented by the specific backend.
 */
bool extent_dalloc(extent_hooks_t *extent_hooks, void *addr, size_t size,
    bool committed, unsigned arena_ind);

/*
 * Custom destroy() extent hook. This is implemented by the specific backend.
 */
void extent_destroy(extent_hooks_t *extent_hooks, void *addr, size_t size,
    bool committed, unsigned arena_ind);

/*
 * Custom alloc() extent hook. This is implemented by the specific backend.
 */
void* device_extent_alloc(extent_hooks_t *extent_hooks, void *new_addr,
    size_t size, size_t alignment, bool *zero, bool *commit,
    unsigned arena_ind);

/*
 * Custom dalloc() extent hook. This is implemented by the specific backend.
 */
bool device_extent_dalloc(extent_hooks_t *extent_hooks, void *addr,
    size_t size, bool committed, unsigned arena_ind);

/*
 * Custom destroy() extent hook. This is implemented by the specific backend.
 */
void device_extent_destroy(extent_hooks_t *extent_hooks, void *addr,
    size_t size, bool committed, unsigned arena_ind);

/*
 * Custom alloc() extent hook.
 */
void* host_extent_alloc(extent_hooks_t *extent_hooks, void *new_addr,
    size_t size, size_t alignment, bool *zero, bool *commit,
    unsigned arena_ind);

/*
 * Custom dalloc() extent hook.
 */
bool host_extent_dalloc(extent_hooks_t *extent_hooks, void *addr,
    size_t size, bool committed, unsigned arena_ind);

/*
 * Custom destroy() extent hook.
 */
void host_extent_destroy(extent_hooks_t *extent_hooks, void *addr,
    size_t size, bool committed, unsigned arena_ind);

}
