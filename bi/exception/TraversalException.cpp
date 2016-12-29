/**
 * @file
 */
#include "bi/exception/TraversalException.hpp"

#include "bi/io/bih_ostream.hpp"

#include <sstream>

bi::TraversalException::TraversalException(const Expression* expr) {
  std::stringstream base;
  bih_ostream buf(base);
  if (expr->loc) {
    buf << expr->loc;
  }
  buf << "error: '.' used with non-model type\n";
  msg = base.str();
}
