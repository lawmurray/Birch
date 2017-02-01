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
  buf << "error: '.' used with non-model type '" << expr->type << "' on left\n";
  msg = base.str();
}
