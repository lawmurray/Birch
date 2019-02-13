/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"
#include "libbirch/thread.hpp"

#include <atomic>

namespace bi {
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
  /**
   * Constructor.
   */
  Buffer();

  /**
   * Copy constructor.
   */
  Buffer(const Buffer<T>& o) = delete;

  /**
   * Assignment operator.
   */
  Buffer<T>& operator=(const Buffer<T>&) = delete;

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
  static int64_t size(const int64_t n);

  /**
   * Id of the thread that allocated the buffer.
   */
  unsigned tid;

private:
  /**
   * Use count (the number of arrays sharing this buffer).
   */
  std::atomic<unsigned> useCount;

  /**
   * First element in the buffer. Taking the address of this gives a pointer
   * to the start of the overallocated buffer.
   */
  alignas(T) char first;
};
}

template<class T>
bi::Buffer<T>::Buffer() :
    tid(bi::tid),
    useCount(0) {
  //
}

template<class T>
void bi::Buffer<T>::incUsage() {
  ++useCount;
}

template<class T>
unsigned bi::Buffer<T>::decUsage() {
  assert(useCount > 0);
  return --useCount;
}

template<class T>
unsigned bi::Buffer<T>::numUsage() const {
  return useCount.load();
}

template<class T>
T* bi::Buffer<T>::buf() {
  return (T*)&first;
}

template<class T>
const T* bi::Buffer<T>::buf() const {
  return (const T*)&first;
}

template<class T>
int64_t bi::Buffer<T>::size(const int64_t n) {
  return n > 0 ? sizeof(T)*n + sizeof(Buffer<T>) - 1u : 0;
  // ^ -1 because `first` field is actually the first byte of the contents
}
