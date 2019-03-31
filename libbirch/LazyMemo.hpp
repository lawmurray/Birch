/**
 * @file
 */
#pragma once
#if ENABLE_LAZY_DEEP_CLONE

#include "libbirch/Counted.hpp"
#include "libbirch/LazyAny.hpp"
#include "libbirch/SharedPtr.hpp"
#include "libbirch/Map.hpp"
#include "libbirch/Set.hpp"

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
   * Is the given memo an ancestor of this?
   */
  bool hasAncestor(LazyMemo* memo);

  /**
   * Is there a parent memo?
   */
  bool hasParent() const;

  /**
   * Get the parent memo.
   */
  LazyMemo* getParent() const;

  /**
   * Fork to create a new child memo for cloning.
   *
   * @return The clone memo.
   */
  LazyMemo* fork();

  /**
   * Run garbage collection sweep on this memo and all ancestors.
   */
  void clean();

  /**
   * Map an object that may not yet have been cloned, cloning it if
   * necessary.
   */
  LazyAny* get(LazyAny* o, LazyMemo* from);

  /**
   * Map an object that may not yet have been cloned, without cloning it.
   * This is used as an optimization for read-only access.
   */
  LazyAny* pull(LazyAny* o, LazyMemo* from);

  /**
   * Map an object that may not yet have been cloned, cloning it and all
   * children if necessary. This can be used to finish a lazy clone.
   */
  LazyAny* finish(LazyAny* o, LazyMemo* from);

  /**
   * Deep mapping of an object through ancestor memos up to the current memo,
   * witout any cloning; get() or pull() should be called on the result to
   * map through this memo.
   *
   * @param o The source object.
   *
   * @return The mapped object.
   */
  LazyAny* source(LazyAny* o, LazyMemo* from);

  /**
   * Shallow copy.
   */
  LazyAny* copy(LazyAny* o);

  /**
   * If this memo is frozen, return the memo to which it should forward,
   * otherwise `this`.
   */
  LazyMemo* getForward();

  /**
   * If this memo is frozen, and the memo to which it should forward has
   * already been created, return that memoo, otherwise `this`.
   */
  LazyMemo* pullForward();

  /**
   * Break cycles that occur in the memo ancestry tree. This should be called
   * on the memo just before its shared reference count is decremented, and
   * must be done manually. The issue is that memos keep a shared pointer
   * to their parents, but also, when frozen, a shared pointer to one of
   * their children (its forwarding memo). This checks if the only remaining
   * shared pointer to this memo is that held by its forwarding memo, and if
   * so deletes the shared reference to that forwarding memo to break the
   * cycle, as it will no longer be required anyway.
   */
  void onDecShared();

protected:
  virtual void doFreeze_();

private:
  /**
   * Parent memo.
   */
  SharedPtr<LazyMemo> parent;

  /**
   * If frozen, memo to which to forward. This must be thread safe, and
   * so an atomic raw pointer is used, with manual shared reference count
   * maintenance.
   */
  std::atomic<LazyMemo*> forward;

  /**
   * Memoization of mappings.
   */
  Map m;

  /**
   * Memoization of ancestry queries.
   */
  Set a;

  /**
   * Generation number (zero is root).
   */
  unsigned gen;
};
}

inline bool libbirch::LazyMemo::hasParent() const {
  return parent;
}

inline libbirch::LazyMemo* libbirch::LazyMemo::getParent() const {
  assert(parent);
  return parent.get();
}

inline libbirch::LazyMemo* libbirch::LazyMemo::fork() {
  return create_(this);
}

inline void libbirch::LazyMemo::clean() {
  m.clean();
}

inline void libbirch::LazyMemo::doFreeze_() {
  m.freeze();
}

#endif
