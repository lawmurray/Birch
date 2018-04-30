/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"

namespace bi {
/**
 * Previous declaration.
 *
 * @ingroup exception
 */
struct PreviousDeclarationException: public CompilerException {
  /**
   * Constructor.
   */
  template<class ParameterType1, class ParameterType2>
  PreviousDeclarationException(ParameterType1* param, ParameterType2* prev);

  /**
   * Constructor.
   */
  template<class ParameterType>
  PreviousDeclarationException(ParameterType* param);
};
}

#include "bi/io/bih_ostream.hpp"

#include <sstream>

template<class ParameterType1, class ParameterType2>
bi::PreviousDeclarationException::PreviousDeclarationException(
    ParameterType1* param, ParameterType2* prev) {
  std::stringstream base;
  bih_ostream buf(base);
  if (param->loc) {
    buf << param->loc;
  }
  buf << "error: already declared\n";
  buf << param << "\n";
  if (prev->loc) {
    buf << prev->loc;
  }
  buf << "note: previous declaration\n";
  buf << prev;
  msg = base.str();
}

template<class ParameterType>
bi::PreviousDeclarationException::PreviousDeclarationException(
    ParameterType* param) {
  std::stringstream base;
  bih_ostream buf(base);
  if (param->loc) {
    buf << param->loc;
  }
  buf << "error: already declared\n";
  if (param->loc) {
    buf << param->loc;
  }
  buf << "note: in\n";
  buf << param << '\n';
  msg = base.str();
}
