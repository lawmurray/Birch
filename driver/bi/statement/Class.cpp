/**
 * @file
 */
#include "bi/statement/Class.hpp"

#include "bi/visitor/all.hpp"

bi::Class::Class(const Annotation annotation, Name* name,
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

bi::Class::~Class() {
  //
}

bool bi::Class::isDeclaration() const {
  return true;
}

bi::Statement* bi::Class::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::Class::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Class::accept(Visitor* visitor) const {
  visitor->visit(this);
}
