/**
 * @file
 */
#include "libbirch/Any.hpp"

#include "libbirch/Label.hpp"

void libbirch::Any::holdLabel() {
  auto label = getLabel();
  assert(label);
  if (label != rootLabel) {
    label->incShared();
  }
}

void libbirch::Any::releaseLabel() {
  auto label = getLabel();
  assert(label);
  if (label != rootLabel) {
    label->decShared();
  }
}
