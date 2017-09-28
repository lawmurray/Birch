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

bool bi::Annotated::isClosed() const {
  return (annotation & IS_CLOSED) > 0;
}
