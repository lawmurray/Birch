/**
 * @file
 */
#pragma once

#include "src/exception/Exception.hpp"

namespace birch {
/**
 * Return type deduction used with non-generic function.
 *
 * @ingroup exception
 */
struct ReturnTypeDeductionException: public Exception {
  /**
   * Constructor.
   */
  template<class T>
  ReturnTypeDeductionException(const T* o);
};
}

#include "src/generate/BirchGenerator.hpp"

template<class T>
birch::ReturnTypeDeductionException::ReturnTypeDeductionException(const T* o) {
  std::stringstream base;
  BirchGenerator buf(base, 0, true);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: return type deduction can only be used with generic functions, or member functions within generic classes\n";
  if (o->loc) {
    buf << o->loc;
  }
  buf << "note: in\n";
  buf << o << '\n';

  msg = base.str();
}
