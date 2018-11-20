/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"
#include "libbirch/Counted.hpp"
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
  //STANDARD_CLONE_FUNCTION
  STANDARD_DESTROY_FUNCTION

  /**
   * Is the given memo an ancestor of this?
   */
  bool hasAncestor(Memo* memo) const;

  /**
   * Shallow mapping of an object that may not yet have been cloned,
   * cloning it if necessary.
   *
   * @param o The source object.
   *
   * @return The mapped object.
   */
  Any* get(Any* o);

  /**
   * Shallow mapping of an object that may not yet have been cloned,
   * without cloning it. This can be used as an optimization for read-only
   * access.
   *
   * @param o The source object.
   *
   * @return The mapped object.
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

private:
  /**
   * Parent memo.
   */
  Memo* parent;

  /**
   * Map of original objects to clones.
   */
  Map clones;
};
}
