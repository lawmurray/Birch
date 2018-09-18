/**
 * @file
 */
#pragma once

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
  /**
   * Default constructor.
   */
  Memo();

  /**
   * Constructor for root.
   */
  Memo(int);

  /**
   * Constructor for clone.
   *
   * @param parent Parent.
   */
  Memo(SharedPtr<Memo> parent);

  /**
   * Destructor.
   */
  virtual ~Memo();

  /**
   * Deallocate.
   */
  virtual void destroy();

  /**
   * Is the given memo an ancestor of this?
   */
  bool hasAncestor(Memo* memo) const;

  /**
   * Shallow retrieval of an object that may not yet have been cloned,
   * cloning it if necessary.
   *
   * @param o Source object.
   *
   * @return The cloned object.
   */
  Any* get(Any* o);

  /**
   * Shallow retrieval of an object that may not yet have been cloned,
   * without cloning it. This can be used as an optimization for read-only
   * access to value types.
   *
   * @param o Source object.
   *
   * @return The mapped object.
   */
  Any* pull(Any* o);

  /**
   * Deep retrieval of an object that may not yet have been cloned,
   * without cloning it. This can be used as an optimization for read-only
   * access to value types.
   *
   * @param o Source object.
   *
   * @return The mapped object.
   */
  Any* deepPull(Any* o);

private:
  /**
   * Parent memo.
   */
  SharedPtr<Memo> parent;

  /**
   * Is this an internal memo? An internal memo is one with one or more
   * children.
   */
  bool internal;

  /**
   * Cloned objects.
   */
  Map clones;
};
}
