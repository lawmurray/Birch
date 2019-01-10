/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"
#include "libbirch/Counted.hpp"
#include "libbirch/SharedPtr.hpp"
#include "libbirch/WeakPtr.hpp"
#include "libbirch/Allocator.hpp"
#include "libbirch/Map.hpp"

namespace bi {
/**
 * Memo for lazy deep cloning of objects.
 *
 * @ingroup libbirch
 */
class Memo: public Counted {
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
  bool hasAncestor(Memo* memo) const;

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
   * Freeze all values in the memo.
   */
  void freeze();

  /**
   * Get the parent memo.
   */
  SharedPtr<Memo> getParent() const;

  /**
   * Shallow mapping of an object that may not yet have been cloned,
   * cloning it if necessary.
   */
  Any* get(Any* o);

  /**
   * Shallow mapping of an object that may not yet have been cloned,
   * without cloning it. This can be used as an optimization for read-only
   * access.
   */
  Any* pull(Any* o);

  /**
   * Deep mapping of an object through ancestor memos up to the current memo,
   * witout any cloning; get() or pull() should be called on the result to
   * map through this memo.
   *
   * @param o The source object.
   *
   * @return The mapped object.
   */
  Any* deep(Any* o);

public:
  /**
   * Map of original objects to clones.
   */
  Map clones;

private:
  /**
   * Parent memo.
   */
  SharedPtr<Memo> parent;
};
}
