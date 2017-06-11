/**
 * @file
 */
#include "bi/exception/AmbiguousReferenceException.hpp"

#include "bi/expression/FuncParameter.hpp"
#include "bi/expression/BinaryParameter.hpp"
#include "bi/expression/FuncReference.hpp"
#include "bi/expression/BinaryReference.hpp"
#include "bi/io/bih_ostream.hpp"

#include <sstream>

template<class ParameterType, class ReferenceType>
bi::AmbiguousReferenceException::AmbiguousReferenceException(
    ReferenceType* ref, std::list<ParameterType*> matches) {
  std::stringstream base;
  bih_ostream buf(base);
  if (ref->loc) {
    buf << ref->loc;
  }
  buf << "error: ambiguous reference '" << ref->name << "'\n";
  buf << ref << '\n';
  for (auto iter = matches.begin(); iter != matches.end(); ++iter) {
    if ((*iter)->loc) {
      buf << (*iter)->loc;
    }
    buf << "note: candidate\n";
    buf << *iter << '\n';
  }
  msg = base.str();
}

template bi::AmbiguousReferenceException::AmbiguousReferenceException(
    FuncReference* ref, std::list<FuncParameter*> matches);
template bi::AmbiguousReferenceException::AmbiguousReferenceException(
    BinaryReference* ref, std::list<BinaryParameter*> matches);
