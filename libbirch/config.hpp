/**
 * @file
 */
#pragma once

/**
 * @def ENABLE_MEMORY_POOL
 *
 * Set to 1 to use the built-in pooled allocator, or 0 to use standard
 * `malloc`/`realloc`/`free`.
 *
 * When performing memory leak checks with `valgrind`, set this to 0.
 * (Incidentally, may also need to disable OpenMP, at least on macOS.)
 */
#ifndef ENABLE_MEMORY_POOL
#define ENABLE_MEMORY_POOL 1
#endif

/**
 * @def ENABLE_LAZY_DEEP_CLONE
 *
 * Set to 1 to use the lazy deep clone strategy, or 0 to use an eager deep
 * clone.
 */
#ifndef ENABLE_LAZY_DEEP_CLONE
#define ENABLE_LAZY_DEEP_CLONE 1
#endif

/**
 * @def ENABLE_CLONE_MEMO
 *
 * Enable memoization of object clones.
 */
#ifndef ENABLE_CLONE_MEMO
#define ENABLE_CLONE_MEMO 1
#endif

/**
 * @def CLONE_MEMO_INITIAL_SIZE
 *
 * Initial allocation size (number of entries) in maps used for object clone
 * memoization.
 */
#ifndef CLONE_MEMO_INITIAL_SIZE
#define CLONE_MEMO_INITIAL_SIZE 64
#endif

/**
 * @def CLONE_MEMO_DELTA
 *
 * Number of generations between deep clone memoizations.
 */
#ifndef CLONE_MEMO_DELTA
#define CLONE_MEMO_DELTA 2
#endif

/**
 * @def ENABLE_ANCESTRY_MEMO
 *
 * Enable memoization of clone generation ancestry queries.
 */
#ifndef ENABLE_ANCESTRY_MEMO
#define ENABLE_ANCESTRY_MEMO 1
#endif

/**
 * @def ANCESTRY_MEMO_INITIAL_SIZE
 *
 * Initial allocation size (number of entries) in sets used for ancestry
 * memoization.
 */
#ifndef ANCESTRY_MEMO_INITIAL_SIZE
#define ANCESTRY_MEMO_INITIAL_SIZE 8
#endif

/**
 * @def ANCESTRY_MEMO_DELTA
 *
 * Number of generations between ancestry memoizations.
 */
#ifndef ANCESTRY_MEMO_DELTA
#define ANCESTRY_MEMO_DELTA 2
#endif
