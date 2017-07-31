/**
 * @file
 */
#include "bi/exception/InvalidCallException.hpp"

#include "bi/io/bih_ostream.hpp"

#include <sstream>

bi::InvalidCallException::InvalidCallException(Call* o) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: invalid function call '" << o << "'\n";
  if (o->single->type->isFunction()) {
    buf << "note: type of parameters is\n";
    buf << o->single->type << "\n";
    buf << "note: type of arguments is\n";
    buf << o->parens->type << "\n";
  } else {
    buf << "note: expression is not of function type:\n";
    buf << o->single->type << "\n";
  }

  msg = base.str();
}

bi::InvalidCallException::InvalidCallException(BinaryCall* o) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: invalid binary operator call '" << o << "'\n";
  buf << "note: type of left operand is " << o->left->type << "\n";
  buf << "note: type of right operand is " << o->right->type << "\n";
  if (o->op) {
    for (auto overload : o->op->target->overloads) {
      if (overload->loc) {
        buf << overload->loc;
      }
      buf << "note: candidate\n";
      buf << overload << '\n';
    }
  }
  msg = base.str();
}

bi::InvalidCallException::InvalidCallException(UnaryCall* o) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: invalid unary operator call '" << o << "'\n";
  buf << "note: type of operand is " << o->single->type << "\n";
  if (o->op) {
    for (auto overload : o->op->target->overloads) {
      if (overload->loc) {
        buf << overload->loc;
      }
      buf << "note: candidate\n";
      buf << overload << '\n';
    }
  }
  msg = base.str();
}
