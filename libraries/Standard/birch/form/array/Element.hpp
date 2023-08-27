/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct ElementOp {
  template<class T, class U>
  static auto eval(const T& x, const U& y) {
    return numbirch::element(birch::eval(x), birch::eval(y));
  }

  template<class G, class T, class U>
  static auto grad1(G&& g, const T& x, const U& y) {
    return numbirch::element_grad1(std::forward<G>(g), birch::eval(x),
        birch::eval(y));
  }

  template<class G, class T, class U>
  static auto grad2(G&& g, const T& x, const U& y) {
    return numbirch::element_grad2(std::forward<G>(g), birch::eval(x),
        birch::eval(y));
  }

  template<class T, class U>
  static constexpr int rows(const T& x, const U& y) {
    return 1;
  }

  template<class T, class U>
  static constexpr int columns(const T& x, const U& y) {
    return 1;
  }

  template<class T, class U, class V>
  static auto eval(const T& x, const U& y, const V& z) {
    return numbirch::element(birch::eval(x), birch::eval(y), birch::eval(z));
  }

  template<class G, class T, class U, class V>
  static auto grad1(G&& g, const T& x, const U& y, const V& z) {
    return numbirch::element_grad1(std::forward<G>(g), birch::eval(x),
        birch::eval(y), birch::eval(z));
  }

  template<class G, class T, class U, class V>
  static auto grad2(G&& g, const T& x, const U& y, const V& z) {
    return numbirch::element_grad2(std::forward<G>(g), birch::eval(x),
        birch::eval(y), birch::eval(z));
  }

  template<class G, class T, class U, class V>
  static auto grad3(G&& g, const T& x, const U& y, const V& z) {
    return numbirch::element_grad3(std::forward<G>(g), birch::eval(x),
        birch::eval(y), birch::eval(z));
  }

  template<class T, class U, class V>
  static constexpr int rows(const T& x, const U& y, const V& z) {
    return 1;
  }

  template<class T, class U, class V>
  static constexpr int columns(const T& x, const U& y, const V& z) {
    return 1;
  }
};

template<argument... Args>
using Element = Form<ElementOp,Args...>;

template<argument T, argument U>
auto element(T&& x, U&& y) {
  return Element<tag_t<T>,tag_t<U>>(std::in_place, std::forward<T>(x),
      std::forward<U>(y));
}

template<argument T, argument U, argument V>
auto element(T&& x, U&& y, V&& z) {
  return Element<tag_t<T>,tag_t<U>,tag_t<V>>(std::in_place,
      std::forward<T>(x), std::forward<U>(y), std::forward<V>(z));
}

}
