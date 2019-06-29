/**
 * @file
 */
#pragma once
#if ENABLE_LAZY_DEEP_CLONE

#include "libbirch/Counted.hpp"
#include "libbirch/LazyAny.hpp"
#include "libbirch/SharedPtr.hpp"
#include "libbirch/Map.hpp"
#include "libbirch/ReadWriteLock.hpp"

namespace libbirch {
/**
 * Memo for lazy deep clones.
 *
 * @ingroup libbirch
 */
class LazyMemo: public Counted {
  friend class List;
public:
  using class_type_ = LazyMemo;

protected:
  /**
   * Constructor for root node.
   */
  LazyMemo();

  /**
   * Constructor for non-root node.
   *
   * @param parent Parent.
   */
  LazyMemo(LazyMemo* parent);

  /**
   * Destructor.
   */
  virtual ~LazyMemo();

public:
  libbirch_create_function_
  libbirch_emplace_function_
  libbirch_destroy_function_

  /**
   * Fork to create a new child memo for cloning.
   *
   * @return The clone memo.
   */
  LazyMemo* fork();

  /**
   * Map an object that may not yet have been cloned, cloning it if
   * necessary.
   */
  LazyAny* get(LazyAny* o);

  /**
   * Finish the cloning of an object.
   */
  LazyAny* finish(LazyAny* o);

  /**
   * Map an object that may not yet have been cloned, without cloning it.
   * This is used as an optimization for read-only access.
   */
  LazyAny* pull(LazyAny* o);

  /**
   * Shallow copy.
   */
  LazyAny* copy(LazyAny* o);

protected:
  virtual void doFreeze_();

private:
  /**
   * Map of object clones.
   */
  Map m;

  /**
   * Lock.
   */
  ReadWriteLock l;
};
}

inline libbirch::LazyMemo* libbirch::LazyMemo::fork() {
  return create_(this);
}

inline void libbirch::LazyMemo::doFreeze_() {
  ///@todo check if this can recursively visit the same memo several times
  l.read();
  m.freeze();
  l.unread();
}

#endif
