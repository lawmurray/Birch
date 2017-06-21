/**
 * @file
 */
#include "bi/exception/MemberException.hpp"
#include "bi/io/bih_ostream.hpp"

#include <sstream>

bi::MemberException::MemberException(const Expression* expr) {
  std::stringstream base;
  bih_ostream buf(base);
  if (expr->loc) {
    buf << expr->loc;
  }
  buf << "error: '.' used with type '" << expr->type << "' on left, which is not a class type\n";
  msg = base.str();
}
