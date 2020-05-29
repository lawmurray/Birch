/**
 * @file
 */
#include "libbirch/Any.hpp"

#include "libbirch/Label.hpp"

void libbirch::Any::holdLabel() {
  if (label && label != rootLabel) {
    label->incUsage();
  }
}

void libbirch::Any::releaseLabel() {
  if (label && label != rootLabel) {
    if (label->decUsage() == 0u) {
      delete label;
    }
  }
}

void libbirch::Any::replaceLabel(Label* label) {
  if (!isDiscarded()) {
    if (label && label != rootLabel) {
      label->incUsage();
    }
    if (this->label && this->label != rootLabel) {
      if (this->label->decUsage() == 0u) {
        delete this->label;
      }
    }
  }
  this->label = label;
}
