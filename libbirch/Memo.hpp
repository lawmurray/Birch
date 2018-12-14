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
   * Constructor. Creates the root memo.
   */
  Memo();

  /**
   * Constructor.
   *
   * @param parent Parent.
   * @param isForwarding Is this the forwarding child of the parent?
   */
  Memo(Memo* parent, const bool isForwarding);

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
   * Fork.
   *
   * @return The clone memo.
   *
   * Forks the memo, creating two children, one for cloning, one for
   * forwarding. Returns the former. The latter may be retrieved with
   * forwardGet().
   */
  Memo* fork();

  /**
   * Forward.
   *
   * @return If the memo has previously been forked, returns the forwarding
   * child, otherwise returns this.
   */
  Memo* forwardGet();

  /**
   * Forward.
   *
   * @return If the memo has previously been forked, returns the child for
   * forwarding, otherwise returns this, subject to optimization. In the
   * special case where the memo has previously been forked, but it has not
   * yet been necessary to clone any objects to the forwarding child, returns
   * this instead, as an optimization to reduce the depth of the memo tree
   * (by increasing its breadth).
   */
  Memo* forwardPull();

  /**
   * Get the parent memo.
   */
  SharedPtr<Memo> getParent() const;

public:
  /**
   * Map of original objects to clones.
   */
  Map clones;

private:
  /**
   * Parent memo if this is a cloning memo. This is a shared pointer, and the
   * parent will have no pointer to this.
   */
  SharedPtr<Memo> cloneParent;

  /**
   * Parent memo if this is a forwarding memo. This is a weak pointer, and
   * the parent will have a shared pointer to this.
   */
  WeakPtr<Memo> forwardParent;

  /**
   * Child for forwarding, if any. This is a shared pointer, and the child
   * will have a weak pointer to this.
   */
  SharedPtr<Memo> forwardChild;

  /**
   * Has this been forked?
   */
  bool isForked;
};
}
