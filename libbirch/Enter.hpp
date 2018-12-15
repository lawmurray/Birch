/**
 * @file
 */
#pragma once

namespace bi {
/**
 * Wrap a pointer dereference with a context switch while this objet remains
 * in scope.
 *
 * @ingroup libbirch
 */
template<class T>
class Enter {
  template<class U> friend class Enter;
public:
  /**
   * Constructor.
   *
   * @param context The context to enter.
   */
  Enter(T* ptr) :
      ptr(ptr),
      prevContext(cloneMemo.get()) {
    cloneMemo = ptr->getContext();
  }

  template<class U>
  Enter(Enter<U> && o) :
      ptr(static_cast<T*>(o.ptr)),
      prevContext(o.prevContext) {
    o.prevContext = nullptr;
  }

  /**
   * Destructor.
   */
  ~Enter() {
    if (prevContext) {
      cloneMemo = prevContext;
    }
  }

  operator T*() {
    return ptr;
  }

  operator T*() const {
    return ptr;
  }

  T* operator->() {
    return ptr;
  }

  const T* operator->() const {
    return ptr;
  }

  T& operator*() {
    return *ptr;
  }

  const T& operator*() const {
    return *ptr;
  }

private:
  /**
   * The pointer.
   */
  T* ptr;

  /**
   * The previous context, to restore once this is destroyed.
   */
  Memo* prevContext;
};
}
