/**
 * @file
 */
#include "bi/exception/InvalidCallException.hpp"

bi::InvalidCallException::InvalidCallException(Call* o) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: invalid call '" << o << "', parameter type is "
      << o->single->type << "\n";
  msg = base.str();
}

bi::InvalidCallException::InvalidCallException(
    OverloadedCall<BinaryOperator>* o) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: invalid call '" << o << "'\n";
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

bi::InvalidCallException::InvalidCallException(
    OverloadedCall<UnaryOperator>* o) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: invalid call '" << o << "'\n";
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
