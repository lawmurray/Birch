/**
 * @file
 */
#pragma once

#include "birch/form/Base.hpp"

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
      if constexpr (!numbirch::numeric<std::tuple_element_t<2,std::tuple<Args...>>>) {
        auto& arg = std::get<2>(this->tup);
        if (!is_constant(arg)) {
          shallow_grad(arg, std::apply([&](const Args&... args) {
                return Op::grad3(g, args...);
              }, this->tup), visitor);
        }
      }
    }
    if constexpr (argc > 1) {
      if constexpr (!numbirch::numeric<std::tuple_element_t<1,std::tuple<Args...>>>) {
        auto& arg = std::get<1>(this->tup);
        if (!is_constant(arg)) {
          shallow_grad(arg, std::apply([&](const Args&... args) {
                return Op::grad2(g, args...);
              }, this->tup), visitor);
        }
      }
    }
    if constexpr (argc > 0) {
      if constexpr (!numbirch::numeric<std::tuple_element_t<0,std::tuple<Args...>>>) {
        auto& arg = std::get<0>(this->tup);
        if (!is_constant(arg)) {
          shallow_grad(arg, std::apply([&](const Args&... args) {
                return Op::grad1(std::forward<G>(g), args...);
              }, this->tup), visitor);
        }
      }
    }
  } 

  int rows() const {
    return std::apply([](const Args&... args) {
        return Op::rows(args...); },
        this->tup);
  }

  int columns() const {
    return std::apply([](const Args&... args) {
        return Op::columns(args...); },
        this->tup);
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
