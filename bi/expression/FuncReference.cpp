/**
 * @file
 */
#include "bi/expression/FuncReference.hpp"

#include "bi/visitor/all.hpp"

#include <vector>
#include <algorithm>
#include <typeinfo>

bi::FuncReference::FuncReference(shared_ptr<Name> name, Expression* parens,
    const FunctionForm form, shared_ptr<Location> loc, FuncParameter* target) :
    Expression(loc),
    Named(name),
    Reference<FuncParameter>(target),
    Formed(parens, form) {
  //
}

bi::FuncReference::FuncReference(Expression* left, shared_ptr<Name> name,
    Expression* right, shared_ptr<Location> loc, FuncParameter* target) :
    Expression(loc),
    Named(name),
    Reference<FuncParameter>(target),
    Formed(new ParenthesesExpression(new ExpressionList(left, right)),
        BINARY_OPERATOR) {
  //
}

bi::FuncReference::~FuncReference() {
  //
}

bi::Expression* bi::FuncReference::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::FuncReference::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::FuncReference::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bi::possibly bi::FuncReference::dispatch(Expression& o) {
  return o.le(*this);
}

#include "bi/io/bih_ostream.hpp"

bi::possibly bi::FuncReference::le(FuncReference& o) {
  /* intersection of possible targets */
  std::vector<FuncParameter*> a1, a2, a3;
  if (target != nullptr) {
    a1.push_back(target);
  }
  if (o.target != nullptr) {
    a2.push_back(o.target);
  }
  a1.insert(a1.end(), alternatives.begin(), alternatives.end());
  a2.insert(a2.end(), o.alternatives.begin(), o.alternatives.end());
  std::sort(a1.begin(), a1.end());
  std::sort(a2.begin(), a2.end());
  std::set_intersection(a1.begin(), a1.end(), a2.begin(), a2.end(),
      std::inserter(a3, a3.end()));

  return *parens <= *o.parens
      && (possibly(target == o.target)
          || (possible && possibly(a3.size() > 0)));
}

bi::possibly bi::FuncReference::le(FuncParameter& o) {
  return *parens <= *o.parens && o.capture(this);
}

bi::possibly bi::FuncReference::le(VarParameter& o) {
  return *type <= *o.type && o.capture(this);
}
