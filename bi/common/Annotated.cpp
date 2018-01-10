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
  return (annotation & IS_CLOSED);
}

bool bi::Annotated::isReadOnly() const {
  return (annotation & IS_READ_ONLY);
}
