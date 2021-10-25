/**
 * @file
 */
#include "src/statement/MemberPhantom.hpp"

#include "src/visitor/all.hpp"

birch::MemberPhantom::MemberPhantom(const Annotation annotation, Name* name,
    Location* loc) :
    Statement(loc),
    Annotated(annotation),
    Named(name) {
  //
}

void birch::MemberPhantom::accept(Visitor* visitor) const {
  visitor->visit(this);
}
