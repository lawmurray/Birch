/**
 * @file
 */
#pragma once

#include "libbirch/global.hpp"

namespace bi {
/**
 * Shared pointer with copy-on-write semantics.
 *
 * @ingroup libbirch
 *
 * @tparam T Type.
 */
template<class T>
class SharedPointer: public SharedPointer<typename super_type<T>::type> {
  friend class WeakPointer<T> ;
public:
  using value_type = T;
  using this_type = SharedPointer<T>;
  using super_type = SharedPointer<typename super_type<T>::type>;
  using root_type = typename super_type::root_type;

  /**
   * Default constructor.
   */
  SharedPointer();

  /**
   * Null constructor.
   */
  SharedPointer(const std::nullptr_t& o);

  /**
   * Constructor from STL shared pointer.
   */
  SharedPointer(const std::shared_ptr<T>& ptr);

  /**
   * Copy constructor.
   */
  SharedPointer(const SharedPointer<T>& o) = default;

  /**
   * Move constructor.
   */
  SharedPointer(SharedPointer<T> && o) = default;

  /**
   * Generic copy constructor.
   */
  template<class U, typename = std::enable_if_t<is_assignable<T,U>::value>>
  SharedPointer(const SharedPointer<U>& o);

  /**
   * Assignment from null pointer.
   */
  SharedPointer<T>& operator=(const std::nullptr_t& o);

  /**
   * Copy assignment.
   */
  SharedPointer<T>& operator=(const SharedPointer<T>& o) = default;

  /**
   * Move assignment.
   */
  SharedPointer<T>& operator=(SharedPointer<T> && o) = default;

  /**
   * Generic assignment.
   */
  template<class U, typename = std::enable_if_t<is_assignable<T,U>::value>>
  SharedPointer<T>& operator=(const SharedPointer<U>& o);

  /**
   * Value assignment.
   */
  template<class U>
  SharedPointer<T>& operator=(const U& o);

  /**
   * User-defined conversions. This allows pointers to be passed as arguments
   * to functions with value type parameters, where the type of the object
   * pointed to has a conversion to the value type.
   *
   * @seealso is_convertible
   */
  template<class U, typename = std::enable_if_t<is_convertible<T,U>::value>>
  operator U() const;

  /**
   * Get the raw pointer.
   */
  T* get() const;

  /**
   * Dynamic cast. Returns `nullptr` if the cast if unsuccessful.
   */
  template<class U>
  SharedPointer<U> dynamic_pointer_cast() const;

  /**
   * Static cast. Undefined if unsuccessful.
   */
  template<class U>
  SharedPointer<U> static_pointer_cast() const;

  /**
   * Dereference.
   */
  T& operator*() const {
    return *get();
  }

  /**
   * Member access.
   */
  T* operator->() const {
    return get();
  }

  /**
   * Call operator.
   */
  template<class ... Args>
  auto operator()(Args ... args) const {
    return (*get())(args...);
  }
};

template<>
class SharedPointer<Any> {
  friend class WeakPointer<Any>;
  friend class WeakPointer<const Any>;
  friend class SharedPointer<const Any>;
public:
  using value_type = Any;
  using this_type = SharedPointer<value_type>;
  using root_type = this_type;

  SharedPointer();
  SharedPointer(const std::nullptr_t& o);
  SharedPointer(const std::shared_ptr<Any>& ptr);
  SharedPointer(const SharedPointer<Any>& o) = default;
  SharedPointer(SharedPointer<Any> && o) = default;
  SharedPointer<Any>& operator=(const std::nullptr_t& o);
  SharedPointer<Any>& operator=(const SharedPointer<Any>& o) = default;
  SharedPointer<Any>& operator=(SharedPointer<Any> && o) = default;

  /**
   * Is the pointer not null?
   */
  bool query() const;

  /**
   * Get the pointer.
   */
  Any* get() const;

  template<class U>
  SharedPointer<U> dynamic_pointer_cast() const;

  template<class U>
  SharedPointer<U> static_pointer_cast() const;

  /**
   * Dereference.
   */
  Any& operator*() const {
    return *get();
  }

  /**
   * Member access.
   */
  Any* operator->() const {
    return get();
  }

protected:
  /**
   * Wrapped smart pointer.
   */
  std::shared_ptr<Any> ptr;
};

template<>
class SharedPointer<const Any> {
  friend class WeakPointer<const Any>;
public:
  using value_type = const Any;
  using this_type = SharedPointer<const Any>;
  using root_type = this_type;

  SharedPointer();
  SharedPointer(const std::nullptr_t& o);
  SharedPointer(const std::shared_ptr<const Any>& ptr);
  SharedPointer(const SharedPointer<const Any>& o) = default;
  SharedPointer(SharedPointer<const Any> && o) = default;
  SharedPointer(const std::shared_ptr<Any>& ptr);
  SharedPointer(const SharedPointer<Any>& o);
  SharedPointer<const Any>& operator=(const std::nullptr_t& o);
  SharedPointer<const Any>& operator=(const SharedPointer<const Any>& o) = default;
  SharedPointer<const Any>& operator=(SharedPointer<const Any> && o) = default;
  SharedPointer<const Any>& operator=(const SharedPointer<Any>& o);

  /**
   * Is the pointer not null?
   */
  bool query() const;

  /**
   * Get the pointer.
   */
  const Any* get() const;

  template<class U>
  SharedPointer<const U> dynamic_pointer_cast() const;

  template<class U>
  SharedPointer<const U> static_pointer_cast() const;

  /**
   * Dereference.
   */
  const Any& operator*() const {
    return *get();
  }

  /**
   * Member access.
   */
  const Any* operator->() const {
    return get();
  }

protected:
  /**
   * Wrapped smart pointer.
   */
  std::shared_ptr<const Any> ptr;
};
}

template<class T>
bi::SharedPointer<T>::SharedPointer() :
    super_type(std::make_shared<T>()) {
  //
}

template<class T>
bi::SharedPointer<T>::SharedPointer(const std::nullptr_t& o) :
    super_type(o) {
  //
}

template<class T>
bi::SharedPointer<T>::SharedPointer(const std::shared_ptr<T>& ptr) :
    super_type(ptr) {
  //
}

template<class T>
template<class U, typename>
bi::SharedPointer<T>::SharedPointer(const SharedPointer<U>& ptr) :
    super_type(ptr) {
  //
}

template<class T>
bi::SharedPointer<T>& bi::SharedPointer<T>::operator=(
    const std::nullptr_t& o) {
  root_type::operator=(o);
  return *this;
}

template<class T>
template<class U, typename >
bi::SharedPointer<T>& bi::SharedPointer<T>::operator=(
    const SharedPointer<U>& o) {
  root_type::operator=(o);
  return *this;
}

template<class T>
template<class U>
bi::SharedPointer<T>& bi::SharedPointer<T>::operator=(const U& o) {
  *get() = o;
  return *this;
}

template<class T>
template<class U, typename >
bi::SharedPointer<T>::operator U() const {
  return static_cast<U>(*get());
}

template<class T>
T* bi::SharedPointer<T>::get() const {
#ifndef NDEBUG
  return dynamic_cast<T*>(root_type::get());
#else
  return static_cast<T*>(root_type::get());
#endif
}

template<class T>
template<class U>
bi::SharedPointer<U> bi::SharedPointer<T>::dynamic_pointer_cast() const {
  return SharedPointer<U>(std::dynamic_pointer_cast<U>(this->ptr));
}

template<class T>
template<class U>
bi::SharedPointer<U> bi::SharedPointer<T>::static_pointer_cast() const {
  return SharedPointer<U>(std::static_pointer_cast<U>(this->ptr));
}

template<class U>
bi::SharedPointer<U> bi::SharedPointer<bi::Any>::dynamic_pointer_cast() const {
  return SharedPointer<U>(std::dynamic_pointer_cast<U>(ptr));
}

template<class U>
bi::SharedPointer<U> bi::SharedPointer<bi::Any>::static_pointer_cast() const {
  return SharedPointer<U>(std::static_pointer_cast<U>(ptr));
}

template<class U>
bi::SharedPointer<const U> bi::SharedPointer<const bi::Any>::dynamic_pointer_cast() const {
  return SharedPointer<const U>(std::dynamic_pointer_cast<const U>(ptr));
}

template<class U>
bi::SharedPointer<const U> bi::SharedPointer<const bi::Any>::static_pointer_cast() const {
  return SharedPointer<const U>(std::static_pointer_cast<const U>(ptr));
}
