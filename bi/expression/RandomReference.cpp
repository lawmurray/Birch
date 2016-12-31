/**
 * @file
 */
#include "bi/expression/RandomReference.hpp"

#include "bi/expression/RandomParameter.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::RandomReference::RandomReference(shared_ptr<Name> name,
    shared_ptr<Location> loc, const RandomParameter* target) :
    Expression(loc),
    Named(name),
    Reference(target) {
  //
}

bi::RandomReference::~RandomReference() {
  //
}

bi::Expression* bi::RandomReference::acceptClone(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::RandomReference::acceptModify(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::RandomReference::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::RandomReference::operator<=(Expression& o) {
  try {
    RandomReference& o1 = dynamic_cast<RandomReference&>(o);
    return *type <= *o1.type && (o1.canon(this) || o1.check(this));
  } catch (std::bad_cast e) {
    //
  }
  try {
    RandomParameter& o1 = dynamic_cast<RandomParameter&>(o);
    return *type <= *o1.type && o1.capture(this);
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
    ParenthesesExpression& o1 = dynamic_cast<ParenthesesExpression&>(o);
    return *this <= *o1.expr;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

bool bi::RandomReference::operator==(const Expression& o) const {
  try {
    const RandomReference& o1 = dynamic_cast<const RandomReference&>(o);
    return *type == *o1.type && o1.canon(this);
  } catch (std::bad_cast e) {
    //
  }
  return false;
}
