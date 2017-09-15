/**
 * @file
 */
#include "bi/exception/AmbiguousCallException.hpp"

#include "bi/io/bih_ostream.hpp"

#include <sstream>

bi::AmbiguousCallException::AmbiguousCallException(const Argumented* o,
    const std::list<Parameterised*>& matches) {
  std::stringstream base;
  bih_ostream buf(base);

  auto expr = dynamic_cast<const Expression*>(o);
  assert(expr);
  if (expr->loc) {
    buf << expr->loc;
  }
  buf << "error: ambiguous call\n";
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

  for (auto match : matches) {
    auto stmt = dynamic_cast<Statement*>(match);
    assert(stmt);
    if (stmt->loc) {
      buf << stmt->loc;
    }
    buf << "note: candidate\n";
    buf << stmt << '\n';
  }
  msg = base.str();
}
