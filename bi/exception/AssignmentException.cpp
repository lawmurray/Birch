/**
 * @file
 */
#include "bi/exception/AssignmentException.hpp"

#include "bi/io/bih_ostream.hpp"

bi::AssignmentException::AssignmentException(const Assign* o) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: incompatible type in assignment\n";
  if (o->loc) {
    buf << o->loc;
  }
  buf << "note: in\n";
  buf << o << '\n';

  if (o->left->loc) {
    buf << o->left->loc;
  }
  buf << "note: left type is\n";
  buf << o->left->type << '\n';

  if (o->right->loc) {
    buf << o->right->loc;
  }
  buf << "note: right type is\n";
  buf << o->right->type << '\n';
  msg = base.str();
}
