/**
 * @file
 */
#include "bi/common/Annotated.hpp"

bi::Annotated::Annotated(const Annotation annotation) :
    annotation(annotation) {
  //
}

bi::Annotated::~Annotated() {
  //
}

bool bi::Annotated::has(const Annotation a) const {
  return annotation & a;
}

void bi::Annotated::set(const Annotation a) {
  annotation = (Annotation)(annotation | a);
}
