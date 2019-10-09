/**
 * @file
 */
#include "bi/expression/Call.hpp"

#include "bi/visitor/all.hpp"

template<class ObjectType>
bi::Call<ObjectType>::Call(Expression* single, Expression* args,
    Location* loc) :
    Expression(loc),
    Single<Expression>(single),
    Argumented(args) {
  //
}

template<class ObjectType>
bi::Call<ObjectType>::Call(Expression* single, Location* loc) :
    Expression(loc),
    Single<Expression>(single),
    Argumented(new EmptyExpression()) {
  //
}

template<class ObjectType>
bi::Call<ObjectType>::~Call() {
  //
}

template<class ObjectType>
bi::Expression* bi::Call<ObjectType>::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

template<class ObjectType>
bi::Expression* bi::Call<ObjectType>::accept(Modifier* visitor) {
  return visitor->modify(this);
}

template<class ObjectType>
void bi::Call<ObjectType>::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

template class bi::Call<bi::Unknown>;
//template class bi::Call<bi::Function>;
//template class bi::Call<bi::MemberFunction>;
//template class bi::Call<bi::Fiber>;
//template class bi::Call<bi::MemberFiber>;
//template class bi::Call<bi::LocalVariable>;
//template class bi::Call<bi::MemberVariable>;
//template class bi::Call<bi::GlobalVariable>;
template class bi::Call<bi::UnaryOperator>;
template class bi::Call<bi::BinaryOperator>;
