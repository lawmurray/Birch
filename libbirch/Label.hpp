/**
 * @file
 */
#pragma once

#include "libbirch/Any.hpp"
#include "libbirch/Memo.hpp"

namespace libbirch {
/**
 * Label for bookkeeping lazy deep clones.
 *
 * @ingroup libbirch
 */
class Label final: public Any {
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
   * Destructor.
   */
  virtual ~Label() {
    //
  }

  virtual void finish_() override {
    //
  }

  virtual void freeze_() override {
    //
  }

  virtual Label* copy_(Label* label) const override {
    assert(false);
    return nullptr;
  }

  virtual Label* recycle_(Label* label) override {
    return this;
  }

  virtual void discard_() override {
    //
  }

  virtual void restore_() override {
    //
  }

  /**
   * Update a pointer for writing.
   *
   * @param Smart pointer (Shared, Weak or Init).
   */
  template<class P>
  auto get(P& o)  {
    auto ptr = o.get();
    if (ptr && ptr->isFrozen()) {  // isFrozen a useful guard for performance
      lock.write();
      ptr = o.get();  // reload now that within critical region
      auto old = ptr;
      ptr = static_cast<typename P::value_type*>(mapGet(old));
      if (ptr != old) {
        o.replace(ptr);
      }
      lock.unwrite();
    }
    return ptr;
  }

  /**
   * Update a pointer for reading.
   *
   * @param Smart pointer (Shared, Weak or Init).
   */
  template<class P>
  auto pull(P& o) {
    auto ptr = o.get();
    if (ptr && ptr->isFrozen()) {  // isFrozen a useful guard for performance
      lock.read();
      ptr = o.get();  // reload now that within critical region
      auto old = ptr;
      ptr = static_cast<typename P::value_type*>(mapPull(old));
      if (ptr != old) {
        o.replace(ptr);
      }
      lock.unread();
    }
    return ptr;
  }

private:
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
  ReaderWriterLock lock;
};
}
