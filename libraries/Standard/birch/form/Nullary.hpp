/**
 * @file
 */
#pragma once

#include "birch/form/Memo.hpp"

#define BIRCH_NULLARY_FORM(This, ...) \
  __VA_OPT__(Integer __VA_ARGS__;) \
  \
  MEMBIRCH_STRUCT(This, void) \
  MEMBIRCH_STRUCT_MEMBERS() \
  \
  auto operator->() { \
    return this; \
  } \
  \
  auto operator->() const { \
    return this; \
  } \
  \
  operator auto() const { \
    return value(); \
  } \
  \
  auto operator*() const { \
    return wait(value()); \
  } \
  \
  static constexpr void reset() {} \
  static constexpr void relink(const RelinkVisitor&) {} \
  static constexpr void constant() {} \
  static constexpr bool isConstant() { return true; } \
  static constexpr void args(const ArgsVisitor&) {} \
  static constexpr void deepGrad(const GradVisitor&) {}

#define BIRCH_NULLARY_SIZE(This, ...) \
  int rows() const { \
    return rows(eval()); \
  } \
  \
  int columns() const { \
    return columns(eval()); \
  } \
  \
  int length() const { \
    return length(eval()); \
  } \
  \
  int size() const { \
    return size(eval()); \
  }

#define BIRCH_NULLARY_EVAL(This, f, ...) \
  auto value() const { \
    return numbirch::f(__VA_ARGS__); \
  } \
  \
  auto eval() const { \
    return numbirch::f(__VA_ARGS__); \
  } \
  \
  auto peek() const { \
    return numbirch::f(__VA_ARGS__); \
  } \
  \
  auto move(const MoveVisitor& visitor) const { \
    return numbirch::f(__VA_ARGS__); \
  }

#define BIRCH_NULLARY_GRAD(This, ...) \
  template<class G> \
  static constexpr void shallowGrad(const G& __VA_OPT__(,) __VA_ARGS__, \
      const GradVisitor&) {}

#define BIRCH_NULLARY_CALL(This, f, ...) \
  inline auto f(__VA_OPT__(BIRCH_INT(__VA_ARGS__))) { \
    return This{__VA_ARGS__}; \
  }

#define BIRCH_NULLARY_TYPE(This, ...) \
  template<> \
  struct is_form<This> { \
    static constexpr bool value = true; \
  }; \
  \
  template<> \
  struct tag_s<This> { \
    using type = This; \
  }; \
  \
  template<> \
  struct peg_s<This> { \
    using type = This; \
  };
