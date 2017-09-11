/**
 * @file
 */
#include "bi/exception/ConditionException.hpp"

#include "bi/io/bih_ostream.hpp"

#include <sstream>

bi::ConditionException::ConditionException(const Expression* o) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: incompatible type in condition";
  if (o->loc) {
    buf << o->loc;
  }
  buf << "note: in\n";
  buf << o << '\n';

  buf << "note: condition has type '" << o->type << "', but must be 'Boolean'\n";
  msg = base.str();
}
