/**
 * @file
 */
#include "libbirch/Any.hpp"

#include "libbirch/Label.hpp"

void libbirch::Any::setLabel(Label* label) {
  assert(numShared() > 0u);
  releaseLabel();
  this->label = label;
  holdLabel();
}

void libbirch::Any::releaseLabel() {
  auto label = getLabel();
  if (label) {
    label->decShared();
  }
}
