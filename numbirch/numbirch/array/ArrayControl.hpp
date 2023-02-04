/**
 * @file
 */
#pragma once

#include "numbirch/array/Atomic.hpp"

#include <cassert>
#include <cstdint>
#include <cstddef>

namespace numbirch {
/**
 * @internal
 * 
 * Control block for buffers, handling reference counting and event
 * management.
 * 
 * @ingroup array
 */
class ArrayControl {
public:
  /**
   * Constructor.
   *
   * @param size Buffer size.
   * 
   * The object is initialized with a reference count of one. The caller
   * need not (should not) call incShared().
   */
  ArrayControl(const size_t size);

  /**
   * Copy constructor.
   * 
   * @param o Source object.
   * 
   * The new object is initialized with a reference count of one. The caller
   * need not (should not) call incShared().
   */
  ArrayControl(const ArrayControl& o);

  /**
   * Copy constructor with resized allocation.
   * 
   * @param o Source object.
   * @param size Buffer size, in bytes. This may be less than or greater than
   * the size of @p o.
   * 
   * The object is initialized with a reference count of one. The caller need
   * not (should not) call incShared(). If @p size is greater than that of @p
   * o, the extra bytes are uninitialized. If @p size is less than that of
   * @p o, the extra bytes are truncated.
   */
  ArrayControl(const ArrayControl& o, const size_t size);

  /**
   * Destructor.
   */
  ~ArrayControl();

  /**
   * Reference count.
   */
  int numShared() const {
    return r.load();
  }

  /**
   * Increment the shared reference count.
   */
  void incShared() {
    assert(numShared() > 0);
    r.increment();
  }

  /**
   * Decrement the shared reference count and return the new value.
   */
  int decShared() {
    assert(numShared() > 0);
    return --r;
  }

  /**
   * Have all outstanding reads and writes on the buffer finished?
   */
  bool test();

  /**
   * Reallocate the buffer.
   * 
   * @param size New size, in bytes.
   */
  void realloc(const size_t size);

  /**
   * Buffer.
   */
  void* buf;

  /**
   * Stream of allocation.
   */
  void* streamAlloc;

  /**
   * Stream of last write.
   */
  void* streamWrite;

  /**
   * Size of buffer, in bytes.
   */
  size_t size;

  /**
   * Reference count.
   */
  Atomic<int> r;
};

}
