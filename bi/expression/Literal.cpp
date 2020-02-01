/**
 * @file
 */
#include "bi/expression/Literal.hpp"

#include "bi/visitor/all.hpp"

template<class T1>
bi::Literal<T1>::Literal(const std::string& str, Location* loc) :
    Expression(loc), str(str) {
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

template class bi::Literal<bool>;
template class bi::Literal<int64_t>;
template class bi::Literal<double>;
template class bi::Literal<const char*>;
