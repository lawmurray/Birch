/**
 * @file
 */
#include "src/statement/Struct.hpp"

#include "src/visitor/all.hpp"

birch::Struct::Struct(const Annotation annotation, Name* name,
    Expression* typeParams, Expression* params, Type* base, const bool alias,
    Expression* args, Statement* braces,
    Location* loc) :
    Statement(loc),
    Annotated(annotation),
    Named(name),
    TypeParameterised(typeParams),
    Parameterised(params),
    Based(base, alias),
    Argumented(args),
    Braced(braces) {
  //
}

void birch::Struct::accept(Visitor* visitor) const {
  visitor->visit(this);
}
