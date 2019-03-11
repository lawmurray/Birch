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
   * Is there a parent memo?
   */
  bool hasParent() const;

  /**
   * Get the parent memo.
   */
  LazyMemo* getParent() const;

  /**
   * Shallow mapping of an object that may not yet have been cloned,
   * cloning and forwarding it if necessary.
   */
  LazyAny* get(LazyAny* o, LazyMemo* from);

  /**
   * Shallow mapping of an object that may not yet have been cloned,
   * without cloning it. This can be used as an optimization for read-only
   * access.
   */
  LazyAny* pull(LazyAny* o, LazyMemo* from);

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

protected:
  virtual void doFreeze_();

private:
  /**
   * Parent memo.
   */
  SharedPtr<LazyMemo> parent;

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

inline libbirch::LazyMemo::~LazyMemo() {
  //
}

inline libbirch::LazyMemo* libbirch::LazyMemo::fork() {
  return create_(this);
}

inline void libbirch::LazyMemo::clean() {
  m.clean();
}

inline bool libbirch::LazyMemo::hasParent() const {
  return parent;
}

inline libbirch::LazyMemo* libbirch::LazyMemo::getParent() const {
  assert(parent);
  return parent.get();
}

inline void libbirch::LazyMemo::doFreeze_() {
  m.freeze();
}

#endif
