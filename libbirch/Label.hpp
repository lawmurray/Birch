/**
 * @file
 */
#pragma once

#include "libbirch/Memo.hpp"
#include "libbirch/Shared.hpp"

namespace libbirch {
/**
 * Label for bookkeeping lazy deep clones.
 *
 * @ingroup libbirch
 */
class Label {
public:
  /**
   * Constructor.
   */
  Label();

  /**
   * Copy constructor.
   */
  Label(const Label& o);

  /**
   * New operator.
   */
  void* operator new(std::size_t size) {
    return allocate(size);
  }

  /**
   * Delete operator.
   */
  void operator delete(void* ptr) {
    auto o = static_cast<Label*>(ptr);
    o->deallocate();
  }

  void finish(Label* label) {
    if (!finished.exchange(true)) {
      lock.setRead();
      memo.finish(label);
      lock.unsetRead();
    }
  }

  void freeze() {
    if (!frozen.exchange(true)) {
      lock.setRead();
      memo.freeze();
      lock.unsetRead();
    }
  }

  void thaw() {
    finished.store(false);
    frozen.store(false);
  }

  /**
   * Increment the usage count.
   */
  void incUsage() {
    useCount.increment();
  }

  /**
   * Decrement the usage count.
   *
   * @return Use count.
   */
  unsigned decUsage() {
    assert(useCount.load() > 0u);
    return --useCount;
  }

  /**
   * Usage count.
   */
  unsigned numUsage() const {
    return useCount.load();
  }

  /**
   * Update a smart pointer for writing.
   *
   * @param o Smart pointer (Shared, Weak or Init).
   */
  template<class P>
  auto get(P& o)  {
    auto ptr = o.get();
    if (ptr && ptr->isFrozen()) {  // isFrozen a useful guard for performance
      lock.setWrite();
      ptr = o.get();  // reload now that within critical region
      auto old = ptr;
      ptr = static_cast<typename P::value_type*>(mapGet(old));
      if (ptr != old) {
        /* new objects must always be assigned to a Shared first; it is
         * possible that `ptr` points to a new object, resulting from a copy,
         * so we put it in a Shared first then move assign; if `o` is a
         * Shared this will not result in additional reference count
         * updates, while if it's a Weak or Init this satisfies the
         * requirement to assign to Shared first; in this latter case the
         * object will not be immediate destroyed, as it will necessarily be
         * in a memo and so have a nonzero memo shared count */
        Shared<typename P::value_type> shared(ptr);
        o = std::move(shared);
      }
      lock.unsetWrite();
    }
    return ptr;
  }

  /**
   * Update a smart pointer for reading.
   *
   * @param o Smart pointer (Shared, Weak or Init).
   */
  template<class P>
  auto pull(P& o) {
    auto ptr = o.get();
    if (ptr && ptr->isFrozen()) {  // isFrozen a useful guard for performance
      lock.setRead();
      ptr = o.get();  // reload now that within critical region
      auto old = ptr;
      ptr = static_cast<typename P::value_type*>(mapPull(old));
      if (ptr != old) {
        /* unlike get() above, `ptr` cannot point to a new object here, as
         * mapPull() never copies */
        o.replace(ptr);
      }
      lock.unsetRead();
    }
    return ptr;
  }

  /**
   * Map a raw pointer for writing.
   *
   * @param ptr Raw pointer.
   */
  template<class T>
  auto get(T* ptr)  {
    if (ptr && ptr->isFrozen()) {  // isFrozen a useful guard for performance
      lock.setWrite();
      ptr = static_cast<T*>(mapGet(ptr));
      lock.unsetWrite();
    }
    return ptr;
  }

  /**
   * Map a raw pointer for reading.
   *
   * @param ptr Raw pointer.
   */
  template<class T>
  auto pull(T* ptr) {
    if (ptr && ptr->isFrozen()) {  // isFrozen a useful guard for performance
      lock.setRead();
      ptr = static_cast<T*>(mapPull(ptr));
      lock.unsetRead();
    }
    return ptr;
  }

  /**
   * Id of the thread that allocated the buffer.
   */
  int tid;

private:
  void deallocate() {
    assert(useCount.load() == 0u);
    libbirch::deallocate(this, sizeof(Label), tid);
  }

  /**
   * Map an object that may not yet have been cloned, cloning it if
   * necessary.
   */
  Any* mapGet(Any* o);

  /**
   * Map an object that may not yet have been cloned, without cloning it.
   * This is used as an optimization for read-only access.
   */
  Any* mapPull(Any* o);

  /**
   * Memo that maps source objects to clones.
   */
  Memo memo;

  /**
   * Lock.
   */
  ReadersWriterLock lock;

  /**
   * Use count (the number of objects sharing this label).
   */
  Atomic<unsigned> useCount;

  /**
   * Have all entries in the memo been finished?
   */
  Atomic<bool> finished;

  /**
   * Have all entries in the memo been frozen?
   */
  Atomic<bool> frozen;
};
}
