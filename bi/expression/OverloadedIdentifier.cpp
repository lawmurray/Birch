/**
 * @file
 */
#include "bi/expression/OverloadedIdentifier.hpp"

#include "bi/visitor/all.hpp"

template<class ObjectType>
bi::OverloadedIdentifier<ObjectType>::OverloadedIdentifier(Name* name,
    Type* typeArgs, Location* loc, Overloaded<ObjectType>* target) :
    Expression(loc),
    Named(name),
    TypeArgumented(typeArgs),
    Reference<Overloaded<ObjectType>>(target) {
  //
}

template<class ObjectType>
bi::OverloadedIdentifier<ObjectType>::~OverloadedIdentifier() {
  //
}

template<class ObjectType>
bool bi::OverloadedIdentifier<ObjectType>::isOverloaded() const {
  return true;
}

template<class ObjectType>
bi::FunctionType* bi::OverloadedIdentifier<ObjectType>::resolve(Argumented* o) {
  std::set<ObjectType*> matches;
  this->target->overloads.match(o, matches);
  if (matches.size() > 1) {
    throw AmbiguousCallException(o, matches);
  } else if (matches.size() == 1) {
    auto only = *matches.begin();
    return new FunctionType(only->params->type, only->returnType);
  } else {
    /* check inherited */
    auto iter = this->inherited.begin();
    auto end = this->inherited.end();
    while (iter != end) {
      (*iter)->overloads.match(o, matches);
      if (matches.size() > 1) {
        throw AmbiguousCallException(o, matches);
      } else if (matches.size() == 1) {
        auto only = *matches.begin();
        return new FunctionType(only->params->type, only->returnType);
      }
      ++iter;
    }

    /* error */
    std::list<ObjectType*> available;
    std::copy(this->target->overloads.begin(), this->target->overloads.end(),
        std::back_inserter(available));
    throw CallException(o, available);
  }
}

template<class ObjectType>
bi::Expression* bi::OverloadedIdentifier<ObjectType>::accept(
    Cloner* visitor) const {
  return visitor->clone(this);
}

template<class ObjectType>
bi::Expression* bi::OverloadedIdentifier<ObjectType>::accept(
    Modifier* visitor) {
  return visitor->modify(this);
}

template<class ObjectType>
void bi::OverloadedIdentifier<ObjectType>::accept(Visitor* visitor) const {
  visitor->visit(this);
}

template class bi::OverloadedIdentifier<bi::Unknown>;
template class bi::OverloadedIdentifier<bi::Function>;
template class bi::OverloadedIdentifier<bi::Fiber>;
template class bi::OverloadedIdentifier<bi::MemberFunction>;
template class bi::OverloadedIdentifier<bi::MemberFiber>;
template class bi::OverloadedIdentifier<bi::BinaryOperator>;
template class bi::OverloadedIdentifier<bi::UnaryOperator>;
