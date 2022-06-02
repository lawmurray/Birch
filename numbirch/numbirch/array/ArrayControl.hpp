/**
 * @file
 */
#pragma once

#include "numbirch/array/Atomic.hpp"
#include "numbirch/array/Recorder.hpp"

#include <cassert>

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
   * @param bytes Buffer size.
   * 
   * The object is initialized with a reference count of one. The caller
   * need not (should not) call incShared().
   */
  ArrayControl(const size_t bytes);

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
   * @param bytes Number of bytes to re-allocate. This may be less than or
   * greater than the number of bytes in @p o.
   * 
   * The object is initialized with a reference count of one. The caller need
   * not (should not) call incShared(). If @p bytes is greater than the number
   * of bytes in @p o, the extra bytes are uninitialized. If @p bytes is less
   * than the number of bytes in @p o, the extra bytes are truncated.
   */
  ArrayControl(const ArrayControl& o, const size_t bytes);

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
    return --r;
  }

  /**
   * Have all outstanding reads and writes on the buffer finished?
   */
  bool test();

  /**
   * Extend the buffer.
   * 
   * @param extra Extra number of bytes to allocate.
   * 
   * Uses realloc() internally to attempt to extend the allocation,
   * if necessary.
   */
  void realloc(const size_t bytes);

  /**
   * Get underlying buffer for use in a slice operation.
   * 
   * @return Recorder wrapper around the raw pointers. Pass to data() to
   * return the raw pointer while injecting event recording code via its
   * destructor.
   * 
   * @see Recorder for details.
   */
  template<class T>
  Recorder<T> sliced(const int64_t k) {
    void* evt = nullptr;
    event_join(writeEvt);
    if constexpr (!std::is_const_v<T>) {
      event_join(readEvt);
      evt = writeEvt;
    } else {
      evt = readEvt;
    }
    return Recorder<T>(static_cast<T*>(buf) + k, evt);
  }

  /**
   * Get underlying buffer for use in a dice operation.
   * 
   * @return Raw pointer.
   */
  template<class T>
  T* diced(const int64_t k) {
    event_wait(writeEvt);
    if (!std::is_const_v<T>) {
      event_wait(readEvt);
    }
    return static_cast<T*>(buf) + k;
  }

private:
  /**
   * Buffer.
   */
  void* buf;

  /**
   * Read event.
   */
  void* readEvt;

  /**
   * Write event.
   */
  void* writeEvt;

  /**
   * Size of buffer.
   * 
   * @todo As optional, only needed to optimize free(), could pack into an int
   * so that the whole object packs into 32 bytes, rather than 36 at present.
   */
  size_t bytes;

  /**
   * Reference count.
   */
  Atomic<int> r;
};

}
