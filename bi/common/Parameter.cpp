/**
 * @file
 */
#include "bi/common/Parameter.hpp"

#include "bi/program/Prog.hpp"
#include "bi/statement/Statement.hpp"
#include "bi/expression/Expression.hpp"
#include "bi/type/Type.hpp"

#include <cassert>

template<class Argument>
bi::Parameter<Argument>::Parameter() : arg(nullptr) {
  //
}

template<class Argument>
bi::Parameter<Argument>::~Parameter() {
  //
}

template<class Argument>
bool bi::Parameter<Argument>::capture(Argument* arg) {
  /* pre-condition */
  assert(arg);

  this->arg = arg;

  return true;
}

/*
 * Explicit instantiations.
 */
template class bi::Parameter<bi::Prog>;
template class bi::Parameter<bi::Expression>;
template class bi::Parameter<bi::Statement>;
template class bi::Parameter<bi::Type>;
