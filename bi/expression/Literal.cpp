/**
 * @file
 */
#include "bi/expression/Literal.hpp"

#include "bi/expression/VarParameter.hpp"
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
bi::possibly bi::Literal<T1>::dispatch(Expression& o) {
  return o.le(*this);
}

template<class T1>
bi::possibly bi::Literal<T1>::le(Literal<T1>& o) {
  return possibly(value == o.value) && *type <= *o.type;
}

template<class T1>
bi::possibly bi::Literal<T1>::le(VarParameter& o) {
  return *type <= *o.type && o.capture(this);
}

/*
 * Explicit instantiations.
 */
template class bi::Literal<bool>;
template class bi::Literal<int64_t>;
template class bi::Literal<double>;
template class bi::Literal<const char*>;
