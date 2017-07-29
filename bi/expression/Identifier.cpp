/**
 * @file
 */
#include "bi/expression/Identifier.hpp"

#include "bi/visitor/all.hpp"

template<class ObjectType>
bi::Identifier<ObjectType>::Identifier(shared_ptr<Name> name,
    shared_ptr<Location> loc, ObjectType* target) :
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

template<class ObjectType>
bool bi::Identifier<ObjectType>::dispatchDefinitely(
    const Expression& o) const {
  return o.definitely(*this);
}

template<class ObjectType>
bool bi::Identifier<ObjectType>::definitely(
    const Identifier<ObjectType>& o) const {
  return true;
}

template<class ObjectType>
bool bi::Identifier<ObjectType>::definitely(const Parameter& o) const {
  return !this->target || type->definitely(*o.type);
}

template<class ObjectType>
bool bi::Identifier<ObjectType>::dispatchPossibly(const Expression& o) const {
  return o.possibly(*this);
}

template<class ObjectType>
bool bi::Identifier<ObjectType>::possibly(
    const Identifier<ObjectType>& o) const {
  return true;
}

template<class ObjectType>
bool bi::Identifier<ObjectType>::possibly(const Parameter& o) const {
  return !this->target || type->possibly(*o.type);
}

template class bi::Identifier<bi::Unknown>;
template class bi::Identifier<bi::Parameter>;
template class bi::Identifier<bi::MemberParameter>;
template class bi::Identifier<bi::GlobalVariable>;
template class bi::Identifier<bi::LocalVariable>;
template class bi::Identifier<bi::MemberVariable>;
