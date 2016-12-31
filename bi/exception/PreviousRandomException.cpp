/**
 * @file
 */
#include "bi/exception/PreviousRandomException.hpp"

#include "bi/io/bih_ostream.hpp"

#include <sstream>

bi::PreviousRandomException::PreviousRandomException(
    RandomVariable* random, RandomVariable* prev) {
  std::stringstream base;
  bih_ostream buf(base);
  if (random->loc) {
    buf << random->loc;
  }
  buf << "error: redeclaration of random specification for '" << random->left << "'\n";
  buf << random << '\n';
  if (prev->loc) {
    buf << prev->loc;
  }
  buf << "note: previous declaration\n";
  buf << prev << '\n';
  msg = base.str();
}
