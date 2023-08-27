  /**
   * @file
   */
  #pragma once

  #include "birch/form/Form.hpp"

  namespace birch {

  struct OuterOp {
    template<class T>
    static auto eval(const T& x) {
      return numbirch::outer(birch::eval(x));
    }

    template<class T, class U>
    static auto eval(const T& x, const U& y) {
      return numbirch::outer(birch::eval(x), birch::eval(y));
    }

    template<class G, class T>
    static auto grad1(G&& g, const T& x) {
      return numbirch::outer_grad(std::forward<G>(g), birch::eval(x));
    }

    template<class G, class T, class U>
    static auto grad1(G&& g, const T& x, const U& y) {
      return numbirch::outer_grad1(std::forward<G>(g), birch::eval(x),
          birch::eval(y));
    }

    template<class G, class T, class U>
    static auto grad2(G&& g, const T& x, const U& y) {
      return numbirch::outer_grad2(std::forward<G>(g), birch::eval(x),
          birch::eval(y));
    }

    template<class T>
    static int rows(const T& x) {
      return birch::rows(x);
    }

    template<class T, class U>
    static int rows(const T& x, const U& y) {
      return birch::rows(x);
    }

    template<class T>
    static constexpr int columns(const T& x) {
      return birch::rows(x);
    }

    template<class T, class U>
    static constexpr int columns(const T& x, const U& y) {
      return birch::rows(y);
    }
  };

  template<class... Args>
  using Outer = Form<OuterOp,Args...>;

  template<argument T>
  auto outer(T&& x) {
    return Outer<tag_t<T>>(std::in_place, std::forward<T>(x));
  }

  template<argument T, argument U>
  auto outer(T&& x, U&& y) {
    return Outer<tag_t<T>,tag_t<U>>(std::in_place, std::forward<T>(x),
        std::forward<U>(y));
  }

  }
