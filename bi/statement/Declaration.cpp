/**
 * @file
 */
#include "bi/statement/Declaration.hpp"

#include "bi/visitor/all.hpp"

#include <typeinfo>

template<class T>
bi::Declaration<T>::Declaration(T* param, shared_ptr<Location> loc) :
    Statement(loc),
    param(param) {
  assert(param);
}

template<class T>
inline bi::Declaration<T>::~Declaration() {
  //
}

template<class T>
bi::Statement* bi::Declaration<T>::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

template<class T>
bi::Statement* bi::Declaration<T>::accept(Modifier* visitor) {
  return visitor->modify(this);
}

template<class T>
void bi::Declaration<T>::accept(Visitor* visitor) const {
  visitor->visit(this);
}

template<class T>
bool bi::Declaration<T>::dispatch(Statement& o) {
  return o.le(*this);
}

template<class T>
bool bi::Declaration<T>::le(Declaration<T>& o) {
  return *param <= *o.param;
}

/*
 * Explicit instantiations.
 */
template class bi::Declaration<bi::VarParameter>;
template class bi::Declaration<bi::FuncParameter>;
template class bi::Declaration<bi::ProgParameter>;
template class bi::Declaration<bi::ModelParameter>;
