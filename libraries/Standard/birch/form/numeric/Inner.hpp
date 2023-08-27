  /**
   * @file
   */
  #pragma once

  #include "birch/form/Form.hpp"

  namespace birch {

  struct InnerOp {
    template<class T>
    static auto eval(const T& x) {
      return numbirch::inner(birch::eval(x));
    }

    template<class T, class U>
    static auto eval(const T& x, const U& y) {
      return numbirch::inner(birch::eval(x), birch::eval(y));
    }

    template<class G, class T>
    static auto grad1(G&& g, const T& x) {
      return numbirch::inner_grad(std::forward<G>(g), birch::eval(x));
    }

    template<class G, class T, class U>
    static auto grad1(G&& g, const T& x, const U& y) {
      return numbirch::inner_grad1(std::forward<G>(g), birch::eval(x),
          birch::eval(y));
    }

    template<class G, class T, class U>
    static auto grad2(G&& g, const T& x, const U& y) {
      return numbirch::inner_grad2(std::forward<G>(g), birch::eval(x),
          birch::eval(y));
    }

    template<class T>
    static int rows(const T& x) {
      return birch::columns(x);
    }

    template<class T, class U>
    static int rows(const T& x, const U& y) {
      return birch::columns(x);
    }

    template<class T>
    static constexpr int columns(const T& x) {
      return birch::columns(x);
    }

    template<class T, class U>
    static constexpr int columns(const T& x, const U& y) {
      return birch::columns(y);
    }
  };

  template<class... Args>
  using Inner = Form<InnerOp,Args...>;

  template<argument T>
  auto inner(T&& x) {
    return Inner<tag_t<T>>(std::in_place, std::forward<T>(x));
  }

  template<argument T, argument U>
  auto inner(T&& x, U&& y) {
    return Inner<tag_t<T>,tag_t<U>>(std::in_place, std::forward<T>(x),
        std::forward<U>(y));
  }

}

#include "birch/form/array/Diagonal.hpp"

namespace birch {

template<argument T, argument U>
auto inner(const Diagonal<T,int>& x, U&& y) {
  return std::get<0>(x.tup)*std::forward<U>(y.tup);
}

template<argument T, argument U>
auto inner(T&& x, const Diagonal<U,int>& y) {
  return transpose(std::forward<T>(x))*std::get<0>(y.tup);
}

template<argument T, argument U>
auto inner(const Diagonal<T,int>& x, const Diagonal<U,int>& y) {
  assert(std::get<1>(x.tup) == std::get<1>(y.tup));
  return diagonal(std::get<0>(x.tup)*std::get<0>(y.tup),
      std::get<1>(x.tup));
}

}
