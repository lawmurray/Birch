/**
 * @file
 */
#include "libbirch/Any.hpp"

#include "libbirch/Label.hpp"

void libbirch::Any::holdLabel() {
  if (label && label != rootLabel) {
    label->incShared();
  }
}

void libbirch::Any::releaseLabel() {
  if (label && label != rootLabel) {
    label->decShared();
  }
}

void libbirch::Any::replaceLabel(Label* label) {
  if (!isDiscarded()) {
    if (label && label != rootLabel) {
      label->incShared();
    }
    if (this->label && this->label != rootLabel) {
      this->label->decShared();
    }
  }
  this->label = label;
}
