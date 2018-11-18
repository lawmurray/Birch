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

  /**
   * Create an object,
   */
  template<class... Args>
  static Memo* create(Args... args) {
    return emplace(allocate<sizeof(Memo)>(), args...);
  }

  /**
   * Create an object in previously-allocated memory.
   */
  template<class... Args>
  static Memo* emplace(void* ptr, Args... args) {
    auto o = new (ptr) Memo();
    o->size = sizeof(Memo);
    return o;
  }

  /**
   * Deallocate.
   */
  virtual void destroy() {
    this->~Memo();
  }

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
