/**
 * @file
 */
#include "bi/exception/PreviousDeclarationException.hpp"

#include "bi/expression/VarParameter.hpp"
#include "bi/expression/FuncParameter.hpp"
#include "bi/type/ModelParameter.hpp"
#include "bi/program/ProgParameter.hpp"
#include "bi/io/bih_ostream.hpp"

#include <sstream>

template<class ParameterType>
bi::PreviousDeclarationException::PreviousDeclarationException(
    ParameterType* param, ParameterType* prev) {
  std::stringstream base;
  bih_ostream buf(base);
  if (param->loc) {
    buf << param->loc;
  }
  buf << "error: redeclaration of '" << param->name << "'\n";
  buf << param << '\n';
  if (prev->loc) {
    buf << prev->loc;
  }
  buf << "note: previous declaration\n";
  buf << prev << '\n';
  msg = base.str();
}

template bi::PreviousDeclarationException::PreviousDeclarationException(
    VarParameter* param, VarParameter* prev);
template bi::PreviousDeclarationException::PreviousDeclarationException(
    FuncParameter* param, FuncParameter* prev);
template bi::PreviousDeclarationException::PreviousDeclarationException(
    Dispatcher* param, Dispatcher* prev);
template bi::PreviousDeclarationException::PreviousDeclarationException(
    ModelParameter* param, ModelParameter* prev);
template bi::PreviousDeclarationException::PreviousDeclarationException(
    ProgParameter* param, ProgParameter* prev);
