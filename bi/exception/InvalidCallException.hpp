/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/expression/Call.hpp"
#include "bi/expression/OverloadedCall.hpp"

namespace bi {
/**
 * Invalid function call.
 *
 * @ingroup compiler_exception
 */
struct InvalidCallException: public CompilerException {
  /**
   * Constructor.
   */
  InvalidCallException(Call* o);

  /**
   * Constructor.
   */
  InvalidCallException(OverloadedCall<BinaryOperator>* o);

  /**
   * Constructor.
   */
  InvalidCallException(OverloadedCall<UnaryOperator>* o);

  /**
   * Constructor.
   */
  template<class ObjectType>
  InvalidCallException(OverloadedCall<ObjectType>* o);
};
}

#include "bi/io/bih_ostream.hpp"

#include <sstream>

template<class ObjectType>
bi::InvalidCallException::InvalidCallException(
    OverloadedCall<ObjectType>* o) {
  std::stringstream base;
  bih_ostream buf(base);
  if (o->loc) {
    buf << o->loc;
  }
  buf << "error: invalid call '" << o << "'\n";
  auto single =
      dynamic_cast<OverloadedIdentifier<ObjectType>*>(o->single.get());
  if (single) {
    for (auto overload : single->target->overloads) {
      if (overload->loc) {
        buf << overload->loc;
      }
      buf << "note: candidate\n";
      buf << overload << '\n';
    }
  }
  msg = base.str();
}
