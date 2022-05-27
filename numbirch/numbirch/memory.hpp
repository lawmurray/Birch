/**
 * @file
 */
#pragma once

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
 * Synchronize with the device stream associated with the current thread.
 * 
 * @ingroup memory
 */
void wait();

/**
 * Terminate NumBirch.
 * 
 * @ingroup memory
 */
void term();

/**
 * Record an event in the device stream associated with the current thread.
 * 
 * @ingroup memory
 * 
 * @return A type-erased event handle that may be later passed to wait() or
 * forget().
 */
void* record();

/**
 * Wait on an event.
 * 
 * @ingroup memory
 * 
 * @param evt A type-erased event handle, previously returned by record().
 */
void wait(void* evt);

/**
 * Forget an event.
 * 
 * @ingroup memory
 * 
 * @param evt A type-erased event handle, previously returned by record().
 * 
 * Forgets the event, so that it may no longer be passed to wait().
 */
void forget(void* evt);

/**
 * Allocate memory.
 * 
 * @ingroup memory
 * 
 * @param size Number of bytes to allocate.
 * 
 * @return New allocation.
 * 
 * @attention malloc() may return memory that is still in use but that will be
 * safely available by the time it is used again by NumBirch. To safely use
 * the allocation outside of NumBirch, one should use wait() either before or
 * after the call to malloc() to ensure that the memory is no longer in use.
 * Array handles this for you.
 */
void* malloc(const size_t size);

/**
 * Reallocate memory.
 * 
 * @ingroup memory
 * 
 * @param ptr Existing allocation.
 * @param size New size of allocation.
 * 
 * @return Resized allocation.
 * 
 * @attention If there may be outstanding writes to the existing allocation,
 * one should call wait() before calling realloc(). realloc() may return
 * memory that is still in use but that will be safely available by the time
 * it is used again by NumBirch. To safely use the allocation outside of
 * NumBirch, one should use wait() either before or after the call to
 * realloc() to ensure that the memory is no longer in use. Array handles this
 * for you.
 */
void* realloc(void* ptr, const size_t size);

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
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 * 
 * @param[out] dst Destination.
 * @param dwidth Width of each batch of `dst`, in elements.
 * @param dheight Number of batches of `dst`.
 * @param dpitch Stride between batches of `dst`, in elements.
 * @param src Source.
 * @param spitch Stride between batches of `src`, in elements.
 * @param swidth Width of each batch of `src`, in elements.
 * @param sheight Number of batches of `src`.
 */
template<class T, class U, class = std::enable_if_t<std::is_arithmetic_v<T> &&
    std::is_arithmetic_v<U>,int>>
void memcpy(T* dst, const int dpitch, const U* src, const int spitch,
    const int width, const int height);

/**
 * Set memory by filling with a single value.
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
template<class T, class U, class = std::enable_if_t<
    std::is_arithmetic_v<T>,int>>
void memset(T* dst, const int dpitch, const U value, const int width,
    const int height);

}
