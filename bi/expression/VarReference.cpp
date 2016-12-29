/**
 * @file
 */
#include "bi/expression/VarReference.hpp"

#include "bi/expression/VarParameter.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::VarReference::VarReference(shared_ptr<Name> name,
    shared_ptr<Location> loc, const VarParameter* target) :
    Expression(loc), Named(name), Reference(target) {
  //
}

bi::VarReference::~VarReference() {
  //
}

bi::Expression* bi::VarReference::acceptClone(Cloner* visitor) const {
  return visitor->clone(this);
}

void bi::VarReference::acceptModify(Modifier* visitor) {
  visitor->modify(this);
}

void bi::VarReference::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::VarReference::operator<=(Expression& o) {
  if (!target) {
    /* not yet bound */
    try {
      VarParameter& o1 = dynamic_cast<VarParameter&>(o);
      return o1.capture(this);
    } catch (std::bad_cast e) {
      //
    }
  } else {
    try {
      VarReference& o1 = dynamic_cast<VarReference&>(o);
      return *type <= *o1.type && (o1.canon(this) || o1.check(this));
    } catch (std::bad_cast e) {
      //
    }
    try {
      VarParameter& o1 = dynamic_cast<VarParameter&>(o);
      return *type <= *o1.type && o1.capture(this);
    } catch (std::bad_cast e) {
      //
    }
  }
  try {
    ParenthesesExpression& o1 = dynamic_cast<ParenthesesExpression&>(o);
    return *this <= *o1.expr;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

bool bi::VarReference::operator==(const Expression& o) const {
  try {
    const VarReference& o1 = dynamic_cast<const VarReference&>(o);
    return *type == *o1.type && o1.canon(this);
  } catch (std::bad_cast e) {
    //
  }
  return false;
}
