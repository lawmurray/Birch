/**
 * @file
 */
#include "bi/statement/Declaration.hpp"

#include "bi/visitor/all.hpp"

#include <typeinfo>

template<class T>
bi::Declaration<T>::Declaration(T* param, shared_ptr<Location> loc) :
    Statement(loc), param(param) {
  assert(param);
}

template<class T>
inline bi::Declaration<T>::~Declaration() {
  //
}

template<class T>
bi::Statement* bi::Declaration<T>::acceptClone(Cloner* visitor) const {
  return visitor->clone(this);
}

template<class T>
void bi::Declaration<T>::acceptModify(Modifier* visitor) {
  visitor->modify(this);
}

template<class T>
void bi::Declaration<T>::accept(Visitor* visitor) const {
  visitor->visit(this);
}

template<class T>
bool bi::Declaration<T>::operator<=(Statement& o) {
  try {
    Declaration<T>& o1 = dynamic_cast<Declaration<T>&>(o);
    return *param <= *o1.param;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

template<class T>
bool bi::Declaration<T>::operator==(const Statement& o) const {
  try {
    const Declaration<T>& o1 = dynamic_cast<const Declaration<T>&>(o);
    return *param == *o1.param;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

/*
 * Explicit instantiations.
 */
template class bi::Declaration<bi::VarParameter>;
template class bi::Declaration<bi::FuncParameter>;
template class bi::Declaration<bi::ProgParameter>;
template class bi::Declaration<bi::ModelParameter>;
