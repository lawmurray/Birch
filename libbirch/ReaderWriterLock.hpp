/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"

namespace libbirch {
/**
 * Lock allowing multiple readers but only one writer.
 *
 * @ingroup libbirch
 */
class ReaderWriterLock {
public:
  /**
   * Default constructor.
   */
  ReaderWriterLock();

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

  /**
   * Assuming that the calling thread already has a write lock, downgrades
   * that lock to a read lock.
   */
  void downgrade();

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

inline libbirch::ReaderWriterLock::ReaderWriterLock() :
    readers(0),
    writer(false) {
  //
}

inline void libbirch::ReaderWriterLock::read() {
  readers.increment();
  while (writer.load()) {
    //
  }
}

inline void libbirch::ReaderWriterLock::unread() {
  readers.decrement();
}

inline void libbirch::ReaderWriterLock::write() {
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

inline void libbirch::ReaderWriterLock::unwrite() {
  writer.store(false);
}

inline void libbirch::ReaderWriterLock::downgrade() {
  readers.increment();
  writer.store(false);
}
