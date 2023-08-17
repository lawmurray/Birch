/**
 * @file
 */
#pragma once

#include "numbirch/utility.hpp"

#include <cstddef>
#include <type_traits>

namespace numbirch {
/**
 * Initialize NumBirch. This should be called once at the start of the
 * program. It initializes, for example, thread-local variables necessary for
 * computations.
 * 
 * @ingroup memory
 */
void init();

/**
 * Terminate NumBirch.
 * 
 * @ingroup memory
 */
void term();

/**
 * Allocate memory.
 * 
 * @ingroup memory
 * 
 * @param size Number of bytes to allocate.
 * 
 * @return New allocation.
 */
void* malloc(const size_t size);

/**
 * Reallocate memory.
 * 
 * @ingroup memory
 * 
 * @param oldptr Existing allocation.
 * @param oldsize Old size of allocation.
 * @param newsize New size of allocation.
 * 
 * @return Resized allocation.
 */
void* realloc(void* oldptr, const size_t oldsize, const size_t newsize);

/**
 * Free memory.
 * 
 * @ingroup memory
 * 
 * @param ptr Existing allocation.
 */
void free(void* ptr);

/**
 * Free memory.
 * 
 * @ingroup memory
 * 
 * @param ptr Existing allocation.
 * @param size Size of the existing allocation.
 */
void free(void* ptr, const size_t size);

/**
 * Copy memory.
 * 
 * @ingroup memory
 * 
 * @param[out] dst Destination.
 * @param src Source.
 * @param n Number of bytes.
 */
void memcpy(void* dst, const void* src, size_t n);

/**
 * Copy memory.
 * 
 * @ingroup memory
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * 
 * @param[out] dst Destination.
 * @param dpitch Stride between batches of `dst`, in elements.
 * @param src Source.
 * @param spitch Stride between batches of `src`, in elements.
 * @param width Width of each batch, in elements.
 * @param height Number of batches.
 */
template<arithmetic T, arithmetic U>
NUMBIRCH_KEEP void memcpy(T* dst, const int dpitch, const U* src, const int spitch,
    const int width, const int height);

/**
 * Fill memory with a single value.
 * 
 * @ingroup memory
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * 
 * @param[out] dst Destination.
 * @param dpitch Stride between batches of `dst`, in elements.
 * @param value Value to set.
 * @param width Width of each batch, in elements.
 * @param height Number of batches.
 */
template<arithmetic T, arithmetic U>
NUMBIRCH_KEEP void memset(T* dst, const int dpitch, const U value, const int width,
    const int height);

/**
 * Fill memory with a single value.
 * 
 * @ingroup memory
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * 
 * @param[out] dst Destination.
 * @param dpitch Stride between batches of `dst`, in elements.
 * @param value Value to set.
 * @param width Width of each batch, in elements.
 * @param height Number of batches.
 */
template<arithmetic T, arithmetic U>
NUMBIRCH_KEEP void memset(T* dst, const int dpitch, const U* value, const int width,
    const int height);

/**
 * Get the stream associated with the current thread.
 * 
 * @ingroup memory
 */
void* stream_get();

/**
 * Synchronize the host with a given stream.
 * 
 * @ingroup memory
 * 
 * @param stream Stream.
 */
void stream_wait(void* stream);

/**
 * Synchronize the stream associated the current thread with a given stream.
 * 
 * @ingroup memory
 * 
 * @param stream Stream.
 */
void stream_join(void* stream);

/**
 * Synchronize streams on destruction of an array.
 * 
 * @ingroup memory
 * 
 * @param streamAlloc Stream of allocation of the array.
 * @param stream Stream of last operation involving the array.
 */
void stream_finish(void* streamAlloc, void* stream);

/**
 * Lock the scheduling mutex for exclusive ownership.
 * 
 * @ingroup memory
 */
void lock();

/**
 * Unlock the scheduling mutex for exclusive ownership.
 * 
 * @ingroup memory
 */
void unlock();

/**
 * Lock the scheduling mutex for shared ownership.
 * 
 * @ingroup memory
 */
void lock_shared();

/**
 * Unlock the scheduling mutex for shared ownership.
 * 
 * @ingroup memory
 */
void unlock_shared();

}
