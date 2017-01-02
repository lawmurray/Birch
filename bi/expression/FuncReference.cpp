/**
 * @file
 */
#include "bi/expression/FuncReference.hpp"

#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::FuncReference::FuncReference(shared_ptr<Name> name, Expression* parens,
    const FunctionForm form, shared_ptr<Location> loc,
    const FuncParameter* target) :
    Expression(loc),
    Named(name),
    Reference<FuncParameter>(target),
    Parenthesised(parens),
    Formed(form) {
  //
}

bi::FuncReference::FuncReference(Expression* left, shared_ptr<Name> name,
    Expression* right, shared_ptr<Location> loc, const FuncParameter* target) :
    Expression(loc),
    Named(name),
    Reference<FuncParameter>(target),
    Parenthesised(new ParenthesesExpression(new ExpressionList(left, right))),
    Formed(BINARY_OPERATOR) {
  //
}

bi::FuncReference::~FuncReference() {
  //
}

bi::Expression* bi::FuncReference::acceptClone(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::FuncReference::acceptModify(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::FuncReference::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::FuncReference::operator<=(Expression& o) {
  if (!target) {
    /* not yet bound */
    try {
      FuncParameter& o1 = dynamic_cast<FuncParameter&>(o);
      return *parens <= *o1.parens && o1.capture(this);
    } catch (std::bad_cast e) {
      //
    }
  } else {
    try {
      FuncReference& o1 = dynamic_cast<FuncReference&>(o);
      return *parens <= *o1.parens && *type <= *o1.type
          && (o1.canon(this) || o1.check(this));
    } catch (std::bad_cast e) {
      //
    }
    try {
      FuncParameter& o1 = dynamic_cast<FuncParameter&>(o);
      return *parens <= *o1.parens && *type <= *o1.type && o1.capture(this);
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

bool bi::FuncReference::operator==(const Expression& o) const {
  try {
    const FuncReference& o1 = dynamic_cast<const FuncReference&>(o);
    return *parens == *o1.parens && *type == *o1.type && o1.canon(this);
  } catch (std::bad_cast e) {
    //
  }
  return false;
}
