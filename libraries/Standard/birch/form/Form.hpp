/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

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

#define BIRCH_NO_GRAD \
  template<class G> \
  void shallowGrad(const G& g, const GradVisitor& visitor) { \
    assert(false); \
  }
