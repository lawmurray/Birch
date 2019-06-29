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
class ReadWriteLock {
public:
  /**
   * Default constructor.
   */
  ReadWriteLock();

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

inline libbirch::ReadWriteLock::ReadWriteLock() :
    readers(0),
    writer(false) {
  //
}

inline void libbirch::ReadWriteLock::read() {
  ++readers;
  while (writer.load()) {
    //
  }
}

inline void libbirch::ReadWriteLock::unread() {
  --readers;
}

inline void libbirch::ReadWriteLock::write() {
  bool w;
  do {
    /* obtain the write lock */
    while (writer.exchange(true));

    /* check if there are any readers; if so release the write lock to
     * let those readers proceed and avoid a deadlock situation, repeating
     * from the start, otherwise proceed */
    w = (readers.load() == 0);
    if (!w) {
      writer.store(false);
    }
  } while (!w);
}

inline void libbirch::ReadWriteLock::unwrite() {
  writer.store(false);
}
