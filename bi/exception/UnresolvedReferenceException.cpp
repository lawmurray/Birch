/**
 * @file
 */
#include "bi/exception/UnresolvedReferenceException.hpp"

#include "bi/io/bih_ostream.hpp"

#include <sstream>

template<class ReferenceType>
bi::UnresolvedReferenceException::UnresolvedReferenceException(
    const ReferenceType* ref) {
  std::stringstream base;
  bih_ostream buf(base);
  if (ref->loc) {
    buf << ref->loc;
  }
  buf << "error: unresolved reference '" << ref->name << "'\n";
  if (ref->loc) {
    buf << ref->loc;
  }
  //buf << "note: in\n";
  //buf << ref << '\n';

  msg = base.str();
}

template bi::UnresolvedReferenceException::UnresolvedReferenceException(
    const VarReference* ref);
template bi::UnresolvedReferenceException::UnresolvedReferenceException(
    const FuncReference* ref);
template bi::UnresolvedReferenceException::UnresolvedReferenceException(
    const RandomReference* ref);
template bi::UnresolvedReferenceException::UnresolvedReferenceException(
    const ModelReference* ref);
template bi::UnresolvedReferenceException::UnresolvedReferenceException(
    const ProgReference* ref);
