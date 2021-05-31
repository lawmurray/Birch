/**
 * @file
 */
#pragma once

#include "libbirch/type.hpp"

namespace libbirch {
/**
 * Inplace object.
 *
 * @ingroup libbirch
 *
 * @tparam T Type.
 */
template<class T>
class Inplace {
  friend class Marker;
  friend class Scanner;
  friend class Reacher;
  friend class Collector;
  friend class BiconnectedCollector;
  friend class Spanner;
  friend class Bridger;
  friend class Copier;
  friend class BiconnectedCopier;
  friend class Destroyer;
public:
  using value_type = T;

  /**
   * Default constructor. Constructs a new referent using the default
   * constructor.
   */
  Inplace() = default;

  /**
   * Constructor. Constructs a new object with the given arguments. The
   * first is a placeholder (pass [`std::in_place`](https://en.cppreference.com/w/cpp/utility/in_place))
   * to distinguish this constructor from copy and move constructors.
   * 
   * @note [`std::optional`](https://en.cppreference.com/w/cpp/utility/optional/)
   * behaves similarly with regard to [`std::in_place`](https://en.cppreference.com/w/cpp/utility/in_place).
   */
  template<class... Args>
  Inplace(std::in_place_t, Args&&... args) :
      o(std::forward<Args>(args)...) {
    //
  }

  /**
   * Value assignment.
   */
  template<class U>
  Shared<T>& operator=(const U& o) {
    this->o = o;
    return *this;
  }

  /**
   * Value assignment.
   */
  template<class U>
  const Shared<T>& operator=(const U& o) const {
    return const_cast<Inplace<T>*>(this)->operator=(o);
  }

  /**
   * Dereference.
   */
  T& operator*() {
    return o;
  }

  /**
   * Dereference.
   */
  T& operator*() const {
    return const_cast<Inplace<T>*>(this)->operator*();
  }

  /**
   * Member access.
   */
  T* operator->() {
    return &o;
  }

  /**
   * Member access.
   */
  T* operator->() const {
    return const_cast<Inplace<T>*>(this)->operator->();
  }

  /**
   * Call on referent.
   */
  template<class... Args>
  auto& operator()(Args&&... args) {
    return (*o)(std::forward<Args>(args)...);
  }

  /**
   * Call on referent.
   */
  template<class... Args>
  auto& operator()(Args&&... args) const {
    return const_cast<Inplace<T>*>(this)->operator()(std::forward<Args>(args)...);
  }

private:
  /**
   * Object.
   */
  T o;
};

template<class T>
struct is_inplace<Inplace<T>> {
  static const bool value = true;
};

template<class T>
struct unwrap_inplace<Inplace<T>> {
  using type = T;
};
}
