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
class Label final : public Any {
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
   * Update a smart pointer for writing.
   *
   * @param o Smart pointer (Shared or Init).
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
        o.replace(ptr);
      }
      lock.unsetWrite();
    }
    return ptr;
  }

  /**
   * Update a smart pointer for reading.
   *
   * @param o Smart pointer (Shared or Init).
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
   * Map a raw pointer for reading, with no locking.
   *
   * @param ptr Raw pointer.
   */
  template<class T>
  T* pullNoLock(T* ptr) {
    if (ptr) {
      assert(ptr->isFrozen());
      ptr = static_cast<T*>(mapPull(ptr));
    }
    return ptr;
  }

  /**
   * Copy a raw pointer as first step of deep copy.
   *
   * @param ptr Raw pointer.
   */
  template<class T>
  T* copy(T* ptr)  {
    if (ptr && ptr->isFrozen()) {  // isFrozen a useful guard for performance
      lock.setWrite();
      ptr = static_cast<T*>(mapCopy(ptr));
      lock.unsetWrite();
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
   * Map an object that must be immediately cloned.
   */
  Any* mapCopy(Any* o);

  /**
   * Memo that maps source objects to clones.
   */
  Memo memo;

  /**
   * Lock.
   */
  ReadersWriterLock lock;

public:
  virtual bi::type::String getClassName() const {
    return "Label";
  }

  virtual unsigned size_() const override {
    return sizeof(*this);
  }

  virtual void finish_(libbirch::Label* label) override {
    lock.setRead();
    memo.finish(label);
    lock.unsetRead();
  }

  virtual void freeze_() override {
    lock.setRead();
    memo.freeze();
    lock.unsetRead();
  }

  virtual Label* copy_(libbirch::Label* label) const override {
    return new Label(*this);
  }

  virtual void recycle_(libbirch::Label* label) override {
    //
  }

  virtual void mark_() override {
    memo.mark();
  }

  virtual void scan_() override {
    memo.scan();
  }

  virtual void reach_() override {
    memo.reach();
  }

  virtual void collect_() override {
    memo.collect();
  }

  using base_type = Any;
};

template<unsigned N>
struct is_acyclic_class<Label,N> {
  static const bool value = false;
};
}
