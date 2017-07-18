/**
 * @file
 */
#pragma once

#include <cassert>

namespace bi {
/**
 * Markable, for use with precise garbage collector for coroutine-local
 * allocations.
 */
class Markable {
public:
  /**
   * Constructor.
   */
  Markable() :
      cousage(0),
      index(-1) {
    //
  }

  /**
   * Destructor.
   */
  virtual ~Markable() {
    assert(cousage == 0);
  }

  /**
   * Mark this object as reachable by a coroutine.
   */
  void mark() {
    ++cousage;
  }

  /**
   * Unmark this object as reachable by a coroutine.
   */
  void unmark() {
    assert(cousage > 0);
    --cousage;
  }

  /**
   * Is this object marked by at least one coroutine?
   */
  bool isMarked() const {
    return cousage > 0;
  }

protected:
  /**
   * Create a pointer to this object. This function should be used instead of
   * @c this to obtain a correctly constructed pointer for global or
   * coroutine-local heap allocation, as appropriate for the object. It
   * behaves similarly to @c shared_from_this() in the STL.
   */
  template<class DerivedType>
  Pointer<DerivedType> pointer_from_this() {
    if (index >= 0) {
      return Pointer < DerivedType > (index);
    } else {
      return Pointer<DerivedType>(reinterpret_cast<DerivedType*>(this));
    }
  }
  template<class DerivedType>
  Pointer<const DerivedType> pointer_from_this() const {
    if (index >= 0) {
      return Pointer<const DerivedType>(index);
    } else {
      return Pointer<const DerivedType>(
          reinterpret_cast<DerivedType* const >(this));
    }
  }

private:
  /**
   * Number of coroutines using this object.
   */
  size_t cousage;

  /**
   * Index of this object in its coroutine-local heap, or -1 if on the global
   * heap.
   */
  long index;
};
}
