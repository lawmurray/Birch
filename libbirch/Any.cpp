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
