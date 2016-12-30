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
bi::Expression* bi::Literal<T1>::acceptClone(Cloner* visitor) const {
  return visitor->clone(this);
}

template<class T1>
void bi::Literal<T1>::acceptModify(Modifier* visitor) {
  visitor->modify(this);
}

template<class T1>
void bi::Literal<T1>::accept(Visitor* visitor) const {
  visitor->visit(this);
}

template<class T1>
bool bi::Literal<T1>::operator<=(Expression& o) {
  try {
    const Literal<T1>& o1 = dynamic_cast<const Literal<T1>&>(o);
    return value == o1.value && *type <= *o1.type;
  } catch (std::bad_cast e) {
    //
  }
  try {
    VarParameter& o1 = dynamic_cast<VarParameter&>(o);
    return *type <= *o1.type && o1.capture(this);
  } catch (std::bad_cast e) {
    //
  }
  try {
    VarReference& o1 = dynamic_cast<VarReference&>(o);
    return *type <= *o1.type && o1.check(this);
  } catch (std::bad_cast e) {
    //
  }
  try {
    ParenthesesExpression& o1 = dynamic_cast<ParenthesesExpression&>(o);
    return *this <= *o1.expr;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

template<class T1>
bool bi::Literal<T1>::operator==(const Expression& o) const {
  try {
    const Literal<T1>& o1 = dynamic_cast<const Literal<T1>&>(o);
    return value == o1.value && *type == *o1.type;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

/*
 * Explicit instantiations.
 */
template class bi::Literal<bool>;
template class bi::Literal<int32_t>;
template class bi::Literal<double>;
template class bi::Literal<std::string>;
