/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/statement/Class.hpp"

namespace bi {
/**
 * Invalid generic type arguments.
 *
 * @ingroup exception
 */
struct GenericException: public CompilerException {
  /**
   * Constructor.
   *
   * @param o NamedExpression.
   * @param target Target.
   */
  template<class NamedExpressionType, class ObjectType>
  GenericException(const NamedExpressionType* o, const ObjectType* target);
};
}

#include "bi/io/bih_ostream.hpp"

template<class NamedExpressionType, class ObjectType>
bi::GenericException::GenericException(const NamedExpressionType* o, const ObjectType* target) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: invalid generic type arguments\n";

  if (o->loc) {
    buf << o->loc;
  }
  buf << "note: in\n";
  buf << o << '\n';

  if (target->loc) {
    buf << target->loc;
  }
  buf << "note: target is\n";
  buf << target->name;
  if (!target->typeParams->isEmpty()) {
    buf << '<' << target->typeParams << ">\n";
  }
  msg = base.str();
}
