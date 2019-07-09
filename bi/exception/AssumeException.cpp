/**
 * @file
 */
#include "bi/exception/AssumeException.hpp"

#include "bi/io/bih_ostream.hpp"

bi::AssumeException::AssumeException(const Assume* o) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: incompatible types for probabilistic operator\n";
  if (o->loc) {
    buf << o->loc;
  }
  buf << "note: in\n";
  buf << o;

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
  buf << "note: types should be Value or Random<Value> on left, Distribution<Value> on right, for some common Value type.\n";
  msg = base.str();
}
