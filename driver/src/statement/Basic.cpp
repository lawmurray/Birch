/**
 * @file
 */
#include "src/statement/Basic.hpp"

#include "src/visitor/all.hpp"

birch::Basic::Basic(const Annotation annotation, Name* name,
    Expression* typeParams, Type* base, const bool alias, Location* loc) :
    Statement(loc),
    Annotated(annotation),
    Named(name),
    TypeParameterised(typeParams),
    Based(base, alias) {
  //
}

birch::Statement* birch::Basic::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::Basic::accept(Visitor* visitor) const {
  visitor->visit(this);
}
