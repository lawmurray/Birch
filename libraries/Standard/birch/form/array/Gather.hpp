/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct GatherOp {
  template<class T, class U>
  static auto eval(const T& x, const U& y) {
    return numbirch::gather(birch::eval(x), birch::eval(y));
  }

  template<class G, class T, class U>
  static auto grad1(G&& g, const T& x, const U& y) {
    return numbirch::gather_grad1(std::forward<G>(g), birch::eval(x),
        birch::eval(y));
  }

  template<class G, class T, class U>
  static auto grad2(G&& g, const T& x, const U& y) {
    return numbirch::gather_grad2(std::forward<G>(g), birch::eval(x),
        birch::eval(y));
  }

  template<class T, class U>
  static int rows(const T& x, const U& y) {
    return birch::rows(y);
  }

  template<class T, class U>
  static constexpr int columns(const T& x, const U& y) {
    return 1;
  }

  template<class T, class U, class V>
  static auto eval(const T& x, const U& y, const V& z) {
    return numbirch::gather(birch::eval(x), birch::eval(y), birch::eval(z));
  }

  template<class G, class T, class U, class V>
  static auto grad1(G&& g, const T& x, const U& y, const V& z) {
    return numbirch::gather_grad1(std::forward<G>(g), birch::eval(x),
        birch::eval(y), birch::eval(z));
  }

  template<class G, class T, class U, class V>
  static auto grad2(G&& g, const T& x, const U& y, const V& z) {
    return numbirch::gather_grad2(std::forward<G>(g), birch::eval(x),
        birch::eval(y), birch::eval(z));
  }

  template<class G, class T, class U, class V>
  static auto grad3(G&& g, const T& x, const U& y, const V& z) {
    return numbirch::gather_grad3(std::forward<G>(g), birch::eval(x),
        birch::eval(y), birch::eval(z));
  }

  template<class T, class U, class V>
  static int rows(const T& x, const U& y, const V& z) {
    return birch::rows(y);
  }

  template<class T, class U, class V>
  static int columns(const T& x, const U& y, const V& z) {
    return birch::columns(y);
  }
};

template<argument... Args>
using Gather = Form<GatherOp,Args...>;

template<argument T, argument U>
auto gather(T&& x, U&& y) {
  return Gather<tag_t<T>,tag_t<U>>(std::in_place, std::forward<T>(x),
      std::forward<U>(y));
}

template<argument T, argument U, argument V>
auto gather(T&& x, U&& y, V&& z) {
  return Gather<tag_t<T>,tag_t<U>,tag_t<V>>(std::in_place,
      std::forward<T>(x), std::forward<U>(y), std::forward<V>(z));
}

}
