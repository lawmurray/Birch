/**
 * @file
 */
#include "src/expression/Literal.hpp"

#include "src/visitor/all.hpp"

template<class T1>
birch::Literal<T1>::Literal(const std::string& str, Location* loc) :
    Expression(loc), str(str) {
  //
}

template<class T1>
birch::Expression* birch::Literal<T1>::accept(Modifier* visitor) {
  return visitor->modify(this);
}

template<class T1>
void birch::Literal<T1>::accept(Visitor* visitor) const {
  visitor->visit(this);
}

template class birch::Literal<bool>;
template class birch::Literal<int64_t>;
template class birch::Literal<double>;
template class birch::Literal<const char*>;
