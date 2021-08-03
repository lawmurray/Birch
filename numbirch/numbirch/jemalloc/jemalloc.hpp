/**
 * @file
 * 
 * Jemalloc integration. Implements malloc(), realloc() and free(). Backends
 * using jemalloc should implement extent_alloc(), extent_dalloc() and
 * extent_destroy() instead, as well as call jemalloc_init() during init() and
 * jemalloc_term() during term().
 */
#pragma once

#include <jemalloc/jemalloc.h>

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
 * Allocate for the device arena.
 */
void* device_malloc(const size_t size);

/*
 * Free for the device arena.
 */
void device_free(void* ptr);

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
}
