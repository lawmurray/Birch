/**
 * @file
 */
#pragma once

#include "birch/form/Base.hpp"

#define BIRCH_TRANSFORM_EVAL(f) \
  template<class... Args> \
  static auto eval(const Args&... args) { \
    return numbirch::f(birch::eval(args)...); \
  }

#define BIRCH_TRANSFORM_UNARY_GRAD(grad) \
  template<class G, class... Args> \
  static auto grad1(G&& g, const Args&... args) { \
    return numbirch::grad(std::forward<G>(g), birch::eval(args)...); \
  }

#define BIRCH_TRANSFORM_BINARY_GRAD(grad) \
  template<class G, class... Args> \
  static auto grad1(G&& g, const Args&... args) { \
    return numbirch::grad ## 1(std::forward<G>(g), birch::eval(args)...); \
  } \
  template<class G, class... Args> \
  static auto grad2(G&& g, const Args&... args) { \
    return numbirch::grad ## 2(std::forward<G>(g), birch::eval(args)...); \
  }

#define BIRCH_TRANSFORM_TERNARY_GRAD(grad) \
  template<class G, class... Args> \
  static auto grad1(G&& g, const Args&... args) { \
    return numbirch::grad ## 1(std::forward<G>(g), birch::eval(args)...); \
  } \
  template<class G, class... Args> \
  static auto grad2(G&& g, const Args&... args) { \
    return numbirch::grad ## 2(std::forward<G>(g), birch::eval(args)...); \
  } \
  template<class G, class... Args> \
  static auto grad3(G&& g, const Args&... args) { \
    return numbirch::grad ## 3(std::forward<G>(g), birch::eval(args)...); \
  }

#define BIRCH_TRANSFORM_SIZE \
  template<class... Args> \
  static int rows(const Args&... args) { \
    return birch::rows(args...); \
  } \
  template<class... Args> \
  static int columns(const Args&... args) { \
    return birch::columns(args...); \
  }

namespace birch {

template<class Op, class... Args>
struct Form : public Base<Args...> {
  MEMBIRCH_STRUCT(Form, Base<Args...>)
  MEMBIRCH_STRUCT_MEMBERS()

  template<class... Args1>
  Form(std::in_place_t, Args1&&... args) :
      Base<Args...>(std::in_place, std::forward<Args1>(args)...) {
    //
  }

  template<class... Args1>
  Form(const Form<Op,Args1...>& o) :
      Base<Args...>(o) {
    //
  }

  template<class... Args1>
  Form(Form<Op,Args1...>&& o) :
      Base<Args...>(std::forward<Form<Op,Args1...>>(o)) {
    //
  }

  Form(const Form&) = default;
  Form(Form&&) = default;

  auto operator->() {
    return this;
  }
 
  auto operator->() const {
    return this;
  }
 
  operator auto() const {
    return value();
  }
 
  auto operator*() const {
    return wait(value());
  }

  auto value() const {
    this->constant();
    return eval();
  }

  auto eval() const {
    return std::apply([](const Args&... args) {
        return Op::eval(args...); },
        this->tup);
  }

  template<class G, class T>
  void shallowGrad(G&& g, T&& x, const GradVisitor& visitor) const {
    shallowGrad(std::forward<G>(g), visitor);
  }

  template<class G>
  void shallowGrad(G&& g, const GradVisitor& visitor) const {
    /* work in reverse, so that we can std::forward<G>(g) once regardless
     * of the number of arguments; may also be more cache efficient */
    constexpr int argc = std::tuple_size_v<decltype(this->tup)>;
    if constexpr (argc > 2) {
      auto& arg = std::get<2>(this->tup);
      if (!is_constant(arg)) {
        std::apply([&](const Args&... args) {
              shallow_grad(arg, Op::grad3(g, args...), visitor);
            }, this->tup);
      }
    }
    if constexpr (argc > 1) {
      auto& arg = std::get<1>(this->tup);
      if (!is_constant(arg)) {
        std::apply([&](const Args&... args) {
              shallow_grad(arg, Op::grad2(g, args...), visitor);
            }, this->tup);
      }
    }
    if constexpr (argc > 0) {
      auto& arg = std::get<0>(this->tup);
      if (!is_constant(arg)) {
        std::apply([&](const Args&... args) {
              shallow_grad(arg, Op::grad1(std::forward<G>(g), args...), visitor);
            }, this->tup);
      }
    }
  } 

  template<class Buffer>
  void write(const Buffer& buffer) const {
    buffer->set(value());
  }
 
  template<class Buffer>
  void write(const Integer t, const Buffer& buffer) const {
    buffer->set(value());
  }
};

template<class Op, class... Args>
struct is_form<Form<Op,Args...>> {
  static constexpr bool value = true;
};

template<class Op, class... Args>
struct tag_s<Form<Op,Args...>> {
  using type = Form<Op,tag_t<Args>...>;
};

template<class Op, class... Args>
struct peg_s<Form<Op,Args...>> {
  using type = Form<Op,peg_t<Args>...>;
};

}
