/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"

namespace libbirch {
/**
 * Lock with shared read and exclusive write semantics.
 *
 * @ingroup libbirch
 */
class Lock {
public:
  /**
   * Default constructor.
   */
  Lock();

  /**
   * Obtain read use.
   */
  void read();

  /**
   * Release read use.
   */
  void unread();

  /**
   * Obtain exclusive use.
   */
  void write();

  /**
   * Release exclusive use.
   */
  void unwrite();

private:
  /**
   * Number of readers in critical region.
   */
  Atomic<unsigned> readers;

  /**
   * Is there a writer in the critical region?
   */
  Atomic<bool> writer;
};
}

inline libbirch::Lock::Lock() :
    readers(0),
    writer(false) {
  //
}

inline void libbirch::Lock::read() {
  ++readers;
  while (writer.load()) {
    //
  }
}

inline void libbirch::Lock::unread() {
  --readers;
}

inline void libbirch::Lock::write() {
  bool writer = false;
  while (!writer) {
    /* obtain the write lock */
    do {
      writer = this->writer.exchange(true);
    } while (writer);

    /* check if there are any readers; if so release the write lock to
     * let those readers proceed and avoid a deadlock situation, repeating
     * from the start, otherwise proceed */
    writer = (readers.load() == 0);
    if (!writer) {
      this->writer = false;
    }
  }
}

inline void libbirch::Lock::unwrite() {
  writer = false;
}
