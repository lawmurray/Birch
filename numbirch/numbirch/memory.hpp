/**
 * @file
 * 
 * Memory interface.
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
 * Synchronize with the device. This waits for all operations to complete for
 * the current thread.
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
 * realloc() to ensure that the memory is no longer in use.
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
 * Copy memory.
 * 
 * @ingroup memory
 * 
 * @param[out] dst Destination.
 * @param dpitch Stride between batches of `dst`, in bytes.
 * @param src Source.
 * @param spitch Stride between batches of `src`, in bytes.
 * @param width Width of each batch, in bytes.
 * @param height Number of batches.
 */
void memcpy(void* dst, const size_t dpitch, const void* src,
    const size_t spitch, const size_t width, const size_t height);

/**
 * Set memory by repeating a single value.
 * 
 * @ingroup memory
 * 
 * @tparam T Arithmetic type.
 * 
 * @param[out] dst Destination.
 * @param dpitch Stride between batches of `dst`, in bytes.
 * @param value Value to set.
 * @param width Width of each batch, in bytes.
 * @param height Number of batches.
 */
template<class T, std::enable_if_t<std::is_arithmetic<T>::value,int> = 0>
void memset(void* dst, const size_t dpitch, const T value, const size_t width,
    const size_t height);

}
