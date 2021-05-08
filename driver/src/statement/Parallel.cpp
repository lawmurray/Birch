/**
 * @file
 */
#include "src/statement/Parallel.hpp"

#include "src/visitor/all.hpp"

birch::Parallel::Parallel(const Annotation annotation, Statement* index,
    Expression* from, Expression* to, Statement* braces, Location* loc) :
    Statement(loc),
    Annotated(annotation),
    Braced(braces),
    index(index),
    from(from),
    to(to) {
  //
}

birch::Statement* birch::Parallel::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::Parallel::accept(Visitor* visitor) const {
  visitor->visit(this);
}
