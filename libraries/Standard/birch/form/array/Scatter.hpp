/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct ScatterOp {
  template<class T, class U>
  static auto eval(const T& x, const U& y, const int n) {
    return numbirch::scatter(birch::eval(x), birch::eval(y), n);
  }

  template<class G, class T, class U>
  static auto grad1(G&& g, const T& x, const U& y, const int n) {
    return numbirch::scatter_grad1(std::forward<G>(g), birch::eval(x),
        birch::eval(y), n);
  }

  template<class G, class T, class U>
  static auto grad2(G&& g, const T& x, const U& y, const int n) {
    return numbirch::scatter_grad2(std::forward<G>(g), birch::eval(x),
        birch::eval(y), n);
  }

  template<class T, class U>
  static int rows(const T& x, const U& y, const int n) {
    return n;
  }

  template<class T, class U>
  static constexpr int columns(const T& x, const U& y, const int n) {
    return 1;
  }

  template<class T, class U, class V>
  static auto eval(const T& x, const U& y, const V& z, const int m,
      const int n) {
    return numbirch::scatter(birch::eval(x), birch::eval(y), birch::eval(z),
        m, n);
  }

  template<class G, class T, class U, class V>
  static auto grad1(G&& g, const T& x, const U& y, const V& z, const int m,
      const int n) {
    return numbirch::scatter_grad1(std::forward<G>(g), birch::eval(x),
        birch::eval(y), birch::eval(z), m, n);
  }

  template<class G, class T, class U, class V>
  static auto grad2(G&& g, const T& x, const U& y, const V& z, const int m,
      const int n) {
    return numbirch::scatter_grad2(std::forward<G>(g), birch::eval(x),
        birch::eval(y), birch::eval(z), m, n);
  }

  template<class G, class T, class U, class V>
  static auto grad3(G&& g, const T& x, const U& y, const V& z, const int m,
      const int n) {
    return numbirch::scatter_grad3(std::forward<G>(g), birch::eval(x),
        birch::eval(y), birch::eval(z), m, n);
  }

  template<class T, class U, class V>
  static int rows(const T& x, const U& y, const V& z, const int m,
      const int n) {
    return m;
  }

  template<class T, class U, class V>
  static int columns(const T& x, const U& y, const V& z, const int m,
      const int n) {
    return n;
  }
};

template<argument... Args>
using Scatter = Form<ScatterOp,Args...>;

template<argument T, argument U>
auto scatter(T&& x, U&& y, const int n) {
  return Scatter<tag_t<T>,tag_t<U>,int>(std::in_place, std::forward<T>(x),
      std::forward<U>(y), n);
}

template<argument T, argument U, argument V>
auto scatter(T&& x, U&& y, V&& z, const int m, const int n) {
  return Scatter<tag_t<T>,tag_t<U>,tag_t<V>,int,int>(std::in_place,
      std::forward<T>(x), std::forward<U>(y), std::forward<V>(z), m, n);
}

}
