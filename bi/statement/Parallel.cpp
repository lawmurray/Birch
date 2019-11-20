/**
 * @file
 */
#include "bi/statement/Parallel.hpp"

#include "bi/visitor/all.hpp"

bi::Parallel::Parallel(const Annotation annotation, Statement* index,
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

bi::Parallel::~Parallel() {
  //
}

bi::Statement* bi::Parallel::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::Parallel::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Parallel::accept(Visitor* visitor) const {
  visitor->visit(this);
}
