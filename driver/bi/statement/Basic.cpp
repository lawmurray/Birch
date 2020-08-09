/**
 * @file
 */
#include "bi/statement/Basic.hpp"

#include "bi/visitor/all.hpp"

bi::Basic::Basic(const Annotation annotation, Name* name,
    Expression* typeParams, Type* base, const bool alias, Location* loc) :
    Statement(loc),
    Annotated(annotation),
    Named(name),
    TypeParameterised(typeParams),
    Based(base, alias) {
  //
}

bi::Basic::~Basic() {
  //
}

bi::Statement* bi::Basic::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::Basic::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Basic::accept(Visitor* visitor) const {
  visitor->visit(this);
}
