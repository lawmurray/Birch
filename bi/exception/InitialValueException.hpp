/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/expression/LocalVariable.hpp"
#include "bi/expression/Parameter.hpp"
#include "bi/statement/MemberVariable.hpp"
#include "bi/statement/GlobalVariable.hpp"

namespace bi {
/**
 * Invalid initial value when declaring variable.
 *
 * @ingroup exception
 */
struct InitialValueException: public CompilerException {
  /**
   * Constructor.
   */
  template<class ParameterType>
  InitialValueException(const ParameterType* o);
};
}

#include "bi/io/bih_ostream.hpp"

template<class ParameterType>
bi::InitialValueException::InitialValueException(const ParameterType* o) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: incompatible type in initial value\n";
  if (o->loc) {
    buf << o->loc;
  }
  buf << "note: in\n";
  buf << o << "\n";

  if (o->value->loc) {
    buf << o->value->loc;
  }
  buf << "note: initial value type is\n";
  buf << o->value->type << '\n';
  msg = base.str();
}
