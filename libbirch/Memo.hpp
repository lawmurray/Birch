/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"
#include "libbirch/Counted.hpp"

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
  STANDARD_CLONE_FUNCTION
  STANDARD_DESTROY_FUNCTION

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
