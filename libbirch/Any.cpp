/**
 * @file
 */
#include "libbirch/Any.hpp"

#include "libbirch/Label.hpp"

libbirch::Any* libbirch::Any::clone() const {
  return clone(new Label(getLabel()));
}

libbirch::Any* libbirch::Any::clone(Label* label) const {
  auto o = new Any(*this);
  o->relabel(label);
  return o;
}

libbirch::Any* libbirch::Any::recycle(Label* label) {
  this->relabel(label);
  return this;
}

void libbirch::Any::setLabel(Label* label) {
  assert(numShared() > 0u);
  releaseLabel();
  this->label = label;
  holdLabel();
}

void libbirch::Any::relabel(Label* label) {
  setLabel(label);
}

void libbirch::Any::holdLabel() {
  auto label = getLabel();
  if (label) {
    label->incShared();
  }
}

void libbirch::Any::releaseLabel() {
  auto label = getLabel();
  if (label) {
    label->decShared();
  }
}
