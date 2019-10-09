/**
 * @file
 */
#include "bi/expression/Identifier.hpp"

#include "bi/visitor/all.hpp"

template<class ObjectType>
bi::Identifier<ObjectType>::Identifier(Name* name, Location* loc,
    ObjectType* target) :
    Expression(loc),
    Named(name),
    Reference<ObjectType>(target) {
  //
}

template<class ObjectType>
bi::Identifier<ObjectType>::~Identifier() {
  //
}

template<class ObjectType>
bool bi::Identifier<ObjectType>::isAssignable() const {
  return std::is_same<ObjectType,MemberVariable>::value
  || std::is_same<ObjectType,LocalVariable>::value;
}

template<class ObjectType>
bi::Lookup bi::Identifier<ObjectType>::lookup(Expression* args) {
  return lookup_result<ObjectType>::value;
}

template<class ObjectType>
ObjectType* bi::Identifier<ObjectType>::resolve(Call<ObjectType>* o) {
  if (type->isFunction()) {
    auto functionType = dynamic_cast<FunctionType*>(type);
    assert(functionType);
    if (!o->args->type->isConvertible(*functionType->params)) {
      throw CallException(o, this->target);
    }
  } else {
    throw NotFunctionException(this);
  }
  return this->target;
}

template<class ObjectType>
bi::Expression* bi::Identifier<ObjectType>::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

template<class ObjectType>
bi::Expression* bi::Identifier<ObjectType>::accept(Modifier* visitor) {
  return visitor->modify(this);
}

template<class ObjectType>
void bi::Identifier<ObjectType>::accept(Visitor* visitor) const {
  visitor->visit(this);
}

template class bi::Identifier<bi::Unknown>;
template class bi::Identifier<bi::Parameter>;
template class bi::Identifier<bi::GlobalVariable>;
template class bi::Identifier<bi::LocalVariable>;
template class bi::Identifier<bi::MemberVariable>;
