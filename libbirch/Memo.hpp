/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"
#include "libbirch/Counted.hpp"
#include "libbirch/SharedPtr.hpp"
#include "libbirch/WeakPtr.hpp"

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
   * Forward.
   *
   * @return If the memo has previously been forked, returns the child to
   * which requests should be forwarded, otherwise returns this.
   */
  Memo* forwardGet();

  /**
   * Forward.
   *
   * @return If the memo has previously been forked, returns the child to
   * which requests should be forwarded, otherwise returns this.
   *
   * This version permits an optimization over forwardGet(): if no objects
   * have yet been copied forward, returns this instead.
   */
  Memo* forwardPull();

  /**
   * Fork.
   *
   * @return The clone memo.
   *
   * Forks the memo, creating two children, one for cloning objects, one for
   * forwarding pointers from this. Returns the former. The latter may be
   * retrieved with forward().
   */
  Memo* fork();

public:
  /**
   * Parent memo.
   */
  WeakPtr<Memo> parent;

  /**
   * Child memo to which to forward.
   */
  SharedPtr<Memo> child;

  /**
   * Has this memo been forked?
   */
  bool forked;
};
}
