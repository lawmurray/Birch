/**
 * @file
 */
#include "bi/exception/SequenceException.hpp"

#include "bi/io/bih_ostream.hpp"

bi::SequenceException::SequenceException(const Sequence* o) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: elements of a sequence must have a common type\n";
  if (o->loc) {
    buf << o->loc;
  }
  buf << "note: in\n";
  buf << o << '\n';

  buf << "note: elements have type\n";
  buf << o->single->type << '\n';

  msg = base.str();
}
