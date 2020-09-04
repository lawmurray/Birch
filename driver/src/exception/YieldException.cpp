/**
 * @file
 */
#include "src/exception/YieldException.hpp"

#include "src/generate/BirchGenerator.hpp"

birch::YieldException::YieldException(const Yield* o) {
  std::stringstream base;
  BirchGenerator buf(base, 0, true);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: yield outside fiber.\n";
  if (o->loc) {
    buf << o->loc;
  }
  buf << "note: in\n";
  buf << o;

  msg = base.str();
}
