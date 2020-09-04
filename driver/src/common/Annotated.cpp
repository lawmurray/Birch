/**
 * @file
 */
#include "src/common/Annotated.hpp"

birch::Annotated::Annotated(const Annotation annotation) :
    annotation(annotation) {
  //
}

birch::Annotated::~Annotated() {
  //
}

bool birch::Annotated::has(const Annotation a) const {
  return annotation & a;
}

void birch::Annotated::set(const Annotation a) {
  annotation = (Annotation)(annotation | a);
}
