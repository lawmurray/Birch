/**
 * @file
 */
#include "bi/statement/MemberFiber.hpp"

#include "bi/visitor/all.hpp"

bi::MemberFiber::MemberFiber(const Annotation annotation, Name* name,
    Expression* params, Type* yieldType, Statement* braces, Location* loc) :
    Statement(loc),
    Annotated(annotation),
    Named(name),
    Parameterised(params),
    YieldTyped(yieldType),
    Typed(new EmptyType(loc)),
    Scoped(LOCAL_SCOPE),
    Braced(braces) {
  //
}

bi::MemberFiber::~MemberFiber() {
  //
}

bi::Statement* bi::MemberFiber::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::MemberFiber::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::MemberFiber::accept(Visitor* visitor) const {
  visitor->visit(this);
}
