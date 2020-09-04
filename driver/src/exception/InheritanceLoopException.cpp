/**
 * @file
 */
#include "src/exception/InheritanceLoopException.hpp"

#include "src/generate/BirchGenerator.hpp"

birch::InheritanceLoopException::InheritanceLoopException(const Class* o) {
  std::stringstream base;
  BirchGenerator buf(base, 0, true);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: loop in class inheritance.\n";
  if (o->loc) {
    buf << o->loc;
  }
  buf << "note: in\n";
  buf << o << '\n';

  msg = base.str();
}
