/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {
/**
 * Delayed form. Objects of types derived from `Form` are typically returned
 * by mathematical functions with at least one argument of type
 * [Expression](../../classes/Expression) or, recursively, of a type derived
 * from `Form`.
 */
struct Form {
  MEMBIRCH_STRUCT(Form, MEMBIRCH_NO_BASE)
  MEMBIRCH_STRUCT_MEMBERS(MEMBIRCH_NO_MEMBERS)
};

}

#define BIRCH_FORM_OP \
  auto operator->() { \
    return this; \
  } \
  \
  const auto operator->() const { \
    return this; \
  }

#define BIRCH_FORM \
  int rows() const { \
    return birch::rows(eval()); \
  } \
  \
  int columns() const { \
    return birch::columns(eval()); \
  } \
  \
  int length() const { \
    return birch::length(eval()); \
  } \
  \
  int size() const { \
    return birch::size(eval()); \
  }

#define BIRCH_GRAD_NONE(...) \
  template<class G> \
  void shallowGrad(const G& g, const GradVisitor& visitor) { \
    assert(false); \
  }
