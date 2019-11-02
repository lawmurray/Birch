/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/thread.hpp"
#include "libbirch/Atomic.hpp"

namespace libbirch {
/**
 * Buffer for storing the contents of an array. Buffer objects are shared
 * between arrays with copy-on-write semantics, and overallocated to contain
 * bookkeeping variables, as well as the contents of the buffer itself, in
 * the one allocation. They do not inherit from Countable, as their reference
 * counting semantics are simpler.
 *
 * @ingroup libbirch
 */
template<class T>
class Buffer {
public:
  Buffer(const Buffer& o) = delete;
  Buffer(Buffer&& o) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer& operator=(Buffer&&) = delete;

  /**
   * Constructor.
   */
  Buffer();

  /**
   * Increment the usage count.
   */
  void incUsage();

  /**
   * Decrement the usage count.
   *
   * @return Use count.
   */
  unsigned decUsage();

  /**
   * Usage count.
   */
  unsigned numUsage() const;

  /**
   * Get the start of the buffer.
   */
  T* buf();

  /**
   * Get the start of the buffer.
   */
  const T* buf() const;

  /**
   * Compute the number of bytes that should be allocated for a buffer of
   * this type with @p n elements.
   */
  static size_t size(const int64_t n);

  /**
   * Id of the thread that allocated the buffer.
   */
  int tid;

private:
  /**
   * Use count (the number of arrays sharing this buffer).
   */
  Atomic<unsigned> useCount;

  /**
   * First element in the buffer. Taking the address of this gives a pointer
   * to the start of the overallocated buffer.
   */
  alignas(T) char first;
};
}

template<class T>
libbirch::Buffer<T>::Buffer() :
    tid(get_thread_num()),
    useCount(1) {
  //
}

template<class T>
void libbirch::Buffer<T>::incUsage() {
  useCount.increment();
}

template<class T>
unsigned libbirch::Buffer<T>::decUsage() {
  assert(useCount.load() > 0);
  return --useCount;
}

template<class T>
unsigned libbirch::Buffer<T>::numUsage() const {
  return useCount.load();
}

template<class T>
T* libbirch::Buffer<T>::buf() {
  return (T*)&first;
}

template<class T>
const T* libbirch::Buffer<T>::buf() const {
  return (const T*)&first;
}

template<class T>
size_t libbirch::Buffer<T>::size(const int64_t n) {
  return n > 0 ? sizeof(T)*n + sizeof(Buffer<T>) : 0;
}
