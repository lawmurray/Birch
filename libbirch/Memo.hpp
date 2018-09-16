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
  Memo(const SharedPtr<Memo>& parent);

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
   * Get an object, cloning it if has not been already.
   *
   * @param o The object.
   *
   * @return The cloned object.
   */
  Any* get(Any* o);

  /**
   * Get an object, without cloning it if it has not been already. This is
   * used as an optimization for read-only access.
   *
   * @param o The object.
   *
   * @return The mapped object.
   */
  Any* pull(Any* o);

private:
  /**
   * Parent memo.
   */
  SharedPtr<Memo> parent;

  /**
   * Cloned objects.
   */
  Map clones;

  /**
   * Cache of objects pulled through parent.
   */
  Map cache;
};
}
