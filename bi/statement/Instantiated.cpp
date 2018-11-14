/**
 * @file
 */
#include "bi/statement/Instantiated.hpp"

#include "bi/visitor/all.hpp"

template<class T>
bi::Instantiated<T>::Instantiated(T* single, Location* loc) :
    Statement(loc),
    Single<T>(single) {
  //
}

template<class T>
bi::Instantiated<T>::~Instantiated() {
  //
}

template<class T>
bi::Statement* bi::Instantiated<T>::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

template<class T>
bi::Statement* bi::Instantiated<T>::accept(Modifier* visitor) {
  return visitor->modify(this);
}

template<class T>
void bi::Instantiated<T>::accept(Visitor* visitor) const {
  visitor->visit(this);
}

template class bi::Instantiated<bi::Type>;
template class bi::Instantiated<bi::Expression>;
