/**
 * @file
 */
#include "bi/exception/SuperBaseException.hpp"

#include "bi/io/bih_ostream.hpp"

bi::SuperBaseException::SuperBaseException(const Expression* expr) {
  std::stringstream base;
  bih_ostream buf(base);
  if (expr->loc) {
    buf << expr->loc;
  }
  buf << "error: 'super' used outside of derived class\n";
  msg = base.str();
}
