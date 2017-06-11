/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"

namespace bi {
/**
 * Previous declaration.
 *
 * @ingroup compiler_exception
 */
struct PreviousDeclarationException: public CompilerException {
  /**
   * Constructor.
   */
  template<class ParameterType>
  PreviousDeclarationException(ParameterType* param, ParameterType* prev);
};
}

#include "bi/io/bih_ostream.hpp"

#include <sstream>

template<class ParameterType>
bi::PreviousDeclarationException::PreviousDeclarationException(
    ParameterType* param, ParameterType* prev) {
  std::stringstream base;
  bih_ostream buf(base);
  if (param->loc) {
    buf << param->loc;
  }
  buf << "error: redeclaration of\n";
  buf << param << '\n';
  if (prev->loc) {
    buf << prev->loc;
  }
  buf << "note: previous declaration\n";
  buf << prev << '\n';
  msg = base.str();
}
