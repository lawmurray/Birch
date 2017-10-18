/**
 * @file
 */
#include "bi/exception/CallException.hpp"

#include "bi/io/bih_ostream.hpp"

#include <sstream>

bi::CallException::CallException(Argumented* o,
    const std::list<Parameterised*>& available) {
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
