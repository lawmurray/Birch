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
    Reference<Overloaded<ObjectType>>(target),
    overload(nullptr) {
  //
}

template<class ObjectType>
bi::OverloadedIdentifier<ObjectType>::~OverloadedIdentifier() {
  //
}

template<class ObjectType>
bool bi::OverloadedIdentifier<ObjectType>::isMember() const {
  return std::is_same<ObjectType,MemberFunction>::value ||
      std::is_same<ObjectType,MemberFiber>::value;
}

template<class ObjectType>
bool bi::OverloadedIdentifier<ObjectType>::isOverloaded() const {
  return true;
}

template<class ObjectType>
bi::Expression* bi::OverloadedIdentifier<ObjectType>::resolve(
    Call<Unknown>* o) {
  std::set<ObjectType*> matches;
  this->target->overloads.match(o, matches);

  if (matches.size() > 1) {
    /* try to disambiguate by favouring matches that do not require implicit
     * type conversion; the use of a global variable is a hack */
    allowConversions = false;
    std::set<ObjectType*> preferredMatches;
    is_convertible compare;
    for (auto match : matches) {
      if (compare(o, match)) {
        preferredMatches.insert(match);
      }
    }
    if (preferredMatches.size() == 1) {
      matches = preferredMatches;
    }
    allowConversions = true;
  }

  if (matches.size() > 1) {
    throw AmbiguousCallException(o, matches);
  } else if (matches.size() == 1) {
    overload = *matches.begin();
    return new Call<ObjectType>(o->single, o->args, o->loc, overload);
  } else {
    /* check inherited */
    auto iter = this->inherited.begin();
    auto end = this->inherited.end();
    while (iter != end) {
      (*iter)->overloads.match(o, matches);
      if (matches.size() > 1) {
        throw AmbiguousCallException(o, matches);
      } else if (matches.size() == 1) {
        overload = *matches.begin();
        return new Call<ObjectType>(o->single, o->args, o->loc, overload);
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
