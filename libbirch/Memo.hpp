/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"
#include "libbirch/Counted.hpp"
#include "libbirch/SharedPtr.hpp"
#include "libbirch/Allocator.hpp"
#include "libbirch/Map.hpp"
#include "libbirch/Set.hpp"

namespace bi {
/**
 * Memo for lazy deep cloning of objects.
 *
 * @ingroup libbirch
 */
class Memo: public Counted {
  friend class List;
public:
  using class_type = Memo;

protected:
  /**
   * Constructor.
   *
   * @param parent Parent.
   */
  Memo(Memo* parent = nullptr);

  /**
   * Destructor.
   */
  virtual ~Memo();

public:
  STANDARD_CREATE_FUNCTION
  STANDARD_EMPLACE_FUNCTION
  //STANDARD_FUNCTION
  STANDARD_DESTROY_FUNCTION

  /**
   * Is the given memo an ancestor of this?
   */
  bool hasAncestor(Memo* memo);

  /**
   * Fork to create a new child memo for cloning.
   *
   * @return The clone memo.
   */
  Memo* fork();

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
  Memo* getParent() const;

  /**
   * Shallow mapping of an object that may not yet have been cloned,
   * cloning it if necessary.
   */
  std::pair<Any*,Memo*> get(Any* o, Memo* from);

  /**
   * Shallow mapping of an object that may not yet have been cloned,
   * without cloning it. This can be used as an optimization for read-only
   * access.
   */
  std::pair<Any*,Memo*> pull(Any* o, Memo* from);

  /**
   * Deep mapping of an object through ancestor memos up to the current memo,
   * witout any cloning; get() or pull() should be called on the result to
   * map through this memo.
   *
   * @param o The source object.
   *
   * @return The mapped object.
   */
  Any* source(Any* o, Memo* from);

  /**
   * Shallow copy of an object.
   */
  Any* copy(Any* o);

protected:
  virtual void doFreeze();

private:
  /**
   * Parent memo.
   */
  SharedPtr<Memo> parent;

  /**
   * Memoization of source objects to clones.
   */
  Map m;

  /**
   * Memoization of ancestry queries.
   */
  Set a;
};
}
