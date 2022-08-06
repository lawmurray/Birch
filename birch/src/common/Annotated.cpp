/**
 * @file
 */
#include "src/common/Annotated.hpp"

birch::Annotated::Annotated(const Annotation annotation) :
    annotation(annotation) {
  //
}

bool birch::Annotated::has(const Annotation a) const {
  return (annotation & a) == a;
}

void birch::Annotated::set(const Annotation a) {
  annotation = static_cast<Annotation>(annotation | a);
}
