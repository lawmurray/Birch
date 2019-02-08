/**
 * @file
 */
#pragma once

/**
 * @def USE_MEMORY_POOL
 *
 * Set to 1 to use the built-in pooled allocator, or 0 to use standard
 * `malloc`/`realloc`/`free`.
 *
 * When performing memory leak checks with `valgrind`, set this to 0.
 * (Incidentally, may also need to disable OpenMP, at least on macOS.)
 */
#ifndef USE_MEMORY_POOL
#define USE_MEMORY_POOL 1
#endif

/**
 * @def USE_LAZY_DEEP_CLONE
 *
 * Set to 1 to use the lazy deep clone strategy, or 0 to use an eager deep
 * clone.
 */
#ifndef USE_LAZY_DEEP_CLONE
#define USE_LAZY_DEEP_CLONE 1
#endif

/**
 * @def INITIAL_MAP_SIZE
 *
 * Initial number of entries in each new map used for deep clone memoization.
 */
#ifndef INITIAL_MAP_SIZE
#define INITIAL_MAP_SIZE 64u
#endif

/**
 * @def INITIAL_SET_SIZE
 *
 * Initial number of entries in each new set used for ancestry memoization.
 */
#ifndef INITIAL_SET_SIZE
#define INITIAL_SET_SIZE 8u
#endif
