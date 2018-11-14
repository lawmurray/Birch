/**
 * @file
 */
#pragma once

#include "libbirch/Counted.hpp"

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
  Memo(Memo* parent);

  /**
   * Destructor.
   */
  virtual ~Memo();

  /**
   * Clone.
   */
  Memo* clone() const;

  /**
   * Deallocate.
   */
  virtual void destroy();

  /**
   * Is the given memo an ancestor of this?
   */
  bool hasAncestor(Memo* memo) const;

  /**
   * Get the parent of this memo.
   */
  Memo* getParent();

private:
  /**
   * Parent memo.
   */
  Memo* parent;
};
}
