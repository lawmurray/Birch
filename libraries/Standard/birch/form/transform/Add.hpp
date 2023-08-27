/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct AddOp {
  template<class T, class U>
  static auto eval(const T& x, const U& y) {
    return numbirch::add(birch::eval(x), birch::eval(y));
  }

  template<class G, class T, class U>
  static auto grad1(G&& g, const T& x, const U& y) {
    return numbirch::add_grad1(std::forward<G>(g), birch::eval(x),
        birch::eval(y));
  }

  template<class G, class T, class U>
  static auto grad2(G&& g, const T& x, const U& y) {
    return numbirch::add_grad2(std::forward<G>(g), birch::eval(x),
        birch::eval(y));
  }

  template<class T, class U>
  static int rows(const T& x, const U& y) {
    return birch::rows(x, y);
  }

  template<class T, class U>
  static int columns(const T& x, const U& y) {
    return birch::columns(x, y);
  }
};

template<argument T, argument U>
using Add = Form<AddOp,T,U>;

template<argument T, argument U>
auto add(T&& x, U&& y) {
  return Add<tag_t<T>,tag_t<U>>(std::in_place, std::forward<T>(x),
      std::forward<U>(y));
}

template<argument T, argument U>
requires (!numbirch::arithmetic<T> || !numbirch::arithmetic<U>)
auto operator+(T&& x, U&& y) {
  return add(std::forward<T>(x), std::forward<U>(y));
}

}

#include "birch/form/array/Fill.hpp"

namespace birch {

template<argument T, argument U>
requires (numbirch::arithmetic<decltype(eval(std::declval<U>()))>)
auto operator+(const Fill<T,int>& x, U&& y) {
  return fill(std::get<0>(x.tup) + std::forward<U>(y),
      std::get<1>(x.tup));
}

template<argument T, argument U>
requires (numbirch::arithmetic<decltype(eval(std::declval<T>()))>)
auto operator+(T&& x, const Fill<U,int>& y) {
  return fill(std::forward<T>(x) + std::get<0>(y.tup),
      std::get<1>(y.tup));
}

template<argument T, argument U>
requires (numbirch::arithmetic<decltype(eval(std::declval<U>()))>)
auto operator+(const Fill<T,int,int>& x, U&& y) {
  return fill(std::get<0>(x.tup) + std::forward<U>(y),
      std::get<1>(x.tup), std::get<2>(x.tup));
}

template<argument T, argument U>
requires (numbirch::arithmetic<decltype(eval(std::declval<T>()))>)
auto operator+(T&& x, const Fill<U,int,int>& y) {
  return fill(std::forward<T>(x) + std::get<0>(y.tup),
      std::get<1>(y.tup), std::get<2>(y.tup));
}

}
