/**
 * @file
 */
#include "src/statement/For.hpp"

#include "src/visitor/all.hpp"

birch::For::For(const Annotation annotation, Statement* index,
    Expression* from, Expression* to, Statement* braces, Location* loc) :
    Statement(loc),
    Annotated(annotation),
    Scoped(LOCAL_SCOPE),
    Braced(braces),
    index(index),
    from(from),
    to(to) {
  //
}

birch::For::~For() {
  //
}

birch::Statement* birch::For::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Statement* birch::For::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::For::accept(Visitor* visitor) const {
  visitor->visit(this);
}
