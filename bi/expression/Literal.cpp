/**
 * @file
 */
#include "bi/expression/Literal.hpp"

#include "bi/expression/Parameter.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>
#include <type_traits>
#include <string>

template<class T1>
bi::Literal<T1>::Literal(const T1& value, const std::string& str, Type* type,
    shared_ptr<Location> loc) :
    Expression(type, loc), value(value), str(str) {
  //
}

template<class T1>
bi::Literal<T1>::~Literal() {
  //
}

template<class T1>
bi::Expression* bi::Literal<T1>::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

template<class T1>
bi::Expression* bi::Literal<T1>::accept(Modifier* visitor) {
  return visitor->modify(this);
}

template<class T1>
void bi::Literal<T1>::accept(Visitor* visitor) const {
  visitor->visit(this);
}

template<class T1>
bool bi::Literal<T1>::dispatchDefinitely(const Expression& o) const {
  return o.definitely(*this);
}

template<class T1>
bool bi::Literal<T1>::definitely(Literal<T1>& o) {
  return value == o.value && type->definitely(*o.type);
}

template<class T1>
bool bi::Literal<T1>::definitely(const Parameter& o) const {
  return type->definitely(*o.type);
}

template<class T1>
bool bi::Literal<T1>::dispatchPossibly(const Expression& o) const {
  return o.possibly(*this);
}

template<class T1>
bool bi::Literal<T1>::possibly(Literal<T1>& o) {
  return value == o.value && type->possibly(*o.type);
}

template<class T1>
bool bi::Literal<T1>::possibly(const Parameter& o) const {
  return type->possibly(*o.type);
}

/*
 * Explicit instantiations.
 */
template class bi::Literal<bool>;
template class bi::Literal<int64_t>;
template class bi::Literal<double>;
template class bi::Literal<const char*>;
