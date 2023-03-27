/**
 * @file
 */
#pragma once

#include <cstddef>
#include <type_traits>

namespace numbirch {
class ArrayControl;

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
 * @param oldptr Existing allocation.
 * @param oldsize Old size of allocation.
 * @param newsize New size of allocation.
 * 
 * @return Resized allocation.
 * 
 * realloc() is considered a write to the allocated memory with respect to
 * sequencing and synchronization.
 * 
 * @attention If there may be outstanding writes to the existing allocation,
 * one should call wait() before calling realloc(). realloc() may return
 * memory that is still in use but that will be safely available by the time
 * it is used again by NumBirch. To safely use the allocation outside of
 * NumBirch, one should use wait() either before or after the call to
 * realloc() to ensure that the memory is no longer in use. Array handles this
 * for you.
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
template<class T, class U, class = std::enable_if_t<std::is_arithmetic_v<T> &&
    std::is_arithmetic_v<U>,int>>
void memcpy(T* dst, const int dpitch, const U* src, const int spitch,
    const int width, const int height);

/**
 * Fill memory with a single value.
 * 
 * @ingroup memory
 * 
 * @tparam T Arithmetic type.
 * 
 * @param[out] dst Destination.
 * @param dpitch Stride between batches of `dst`, in elements.
 * @param value Value to set.
 * @param width Width of each batch, in elements.
 * @param height Number of batches.
 */
template<class T, class = std::enable_if_t<std::is_arithmetic_v<T>,int>>
void memset(T* dst, const int dpitch, const T value, const int width,
    const int height);

/**
 * Allocate and initialize an array.
 * 
 * @ingroup memory
 * 
 * @param ctl Control block.
 * @param bytes Number of bytes.
 */
void array_init(ArrayControl* ctl, const size_t bytes);

/**
 * Deallocate an array.
 * 
 * @ingroup memory
 * 
 * @param ctl Control block.
 */
void array_term(ArrayControl* ctl);

/**
 * Resize an array.
 * 
 * @ingroup memory
 * 
 * @param ctl Control block.
 * @param bytes New number of bytes.
 */
void array_resize(ArrayControl* ctl, const size_t bytes);

/**
 * Copy between arrays.
 * 
 * @ingroup memory
 * 
 * @param dst Destination.
 * @param src Source.
 */
void array_copy(ArrayControl* dst, const ArrayControl* src);

/**
 * Wait for completion of reads and writes of an array.
 * 
 * @ingroup memory
 * 
 * @param ctl Control block.
 */
void array_wait(ArrayControl* ctl);

/**
 * Test for completion of reads and writes of an array.
 * 
 * @ingroup memory
 * 
 * @param ctl Control block.
 */
bool array_test(ArrayControl* ctl);

/**
 * Start a read.
 * 
 * @ingroup memory
 * 
 * @param ctl Control block.
 */
void before_read(ArrayControl* ctl);

/**
 * Start a write.
 * 
 * @ingroup memory
 * 
 * @param ctl Control block.
 */
void before_write(ArrayControl* ctl);


/**
 * Finish a read.
 * 
 * @ingroup memory
 * 
 * @param ctl Control block.
 */
void after_read(ArrayControl* ctl);

/**
 * Finish a write.
 * 
 * @ingroup memory
 * 
 * @param ctl Control block.
 */
void after_write(ArrayControl* ctl);

}
