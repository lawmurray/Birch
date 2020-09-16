/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/Atomic.hpp"

namespace libbirch {
/**
 * Lock allowing multiple readers but only one writer.
 *
 * @ingroup libbirch
 */
class ReadersWriterLock {
public:
  /**
   * Default constructor.
   */
  ReadersWriterLock();

  /**
   * Correctly initialize after a bitwise copy.
   */
  void bitwiseFix() {
    readers.store(0u);
    writer.store(false);
  }

  /**
   * Obtain read use.
   */
  void setRead();

  /**
   * Release read use.
   */
  void unsetRead();

  /**
   * Obtain exclusive use.
   */
  void setWrite();

  /**
   * Release exclusive use.
   */
  void unsetWrite();

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

inline libbirch::ReadersWriterLock::ReadersWriterLock() :
    readers(0),
    writer(false) {
  //
}

inline void libbirch::ReadersWriterLock::setRead() {
  readers.increment();
  while (writer.load()) {
    //
  }
}

inline void libbirch::ReadersWriterLock::unsetRead() {
  readers.decrement();
}

inline void libbirch::ReadersWriterLock::setWrite() {
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

inline void libbirch::ReadersWriterLock::unsetWrite() {
  writer.store(false);
}

inline void libbirch::ReadersWriterLock::downgrade() {
  readers.increment();
  writer.store(false);
}
