/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/common/Argumented.hpp"
#include "bi/common/Parameterised.hpp"

namespace bi {
/**
 * Invalid function call.
 *
 * @ingroup exception
 */
struct CallException: public CompilerException {
  /**
   * Constructor.
   *
   * @param o The invalid call.
   * @param available The available overloads.
   */
  template<class ObjectType>
  CallException(Argumented* o,
      const std::list<ObjectType*>& available = std::list<ObjectType*>());
};
}

#include "bi/io/bih_ostream.hpp"

template<class ObjectType>
bi::CallException::CallException(Argumented* o,
    const std::list<ObjectType*>& available) {
  std::stringstream base;
  bih_ostream buf(base);

  auto expr = dynamic_cast<Expression*>(o);
  assert(expr);
  if (expr->loc) {
    buf << expr->loc;
  }
  buf << "error: invalid call\n";
  if (expr->loc) {
    buf << expr->loc;
  }
  buf << "note: in\n";
  buf << expr << "\n";
  if (expr->loc) {
    buf << expr->loc;
  }
  buf << "note: argument types\n";
  buf << o->args->type << "\n";

  for (auto overload : available) {
    auto stmt = dynamic_cast<Statement*>(overload);
    assert(stmt);
    if (stmt->loc) {
      buf << stmt->loc;
    }
    buf << "note: candidate\n";
    buf << stmt;
  }

  msg = base.str();
}
