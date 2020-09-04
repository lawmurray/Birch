/**
 * @file
 */
#include "src/statement/Class.hpp"

#include "src/visitor/all.hpp"

birch::Class::Class(const Annotation annotation, Name* name,
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
    Scoped(MEMBER_SCOPE),
    Braced(braces),
    initScope(new Scope(LOCAL_SCOPE)) {
  //
}

birch::Class::~Class() {
  //
}

bool birch::Class::isDeclaration() const {
  return true;
}

birch::Statement* birch::Class::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Statement* birch::Class::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::Class::accept(Visitor* visitor) const {
  visitor->visit(this);
}
