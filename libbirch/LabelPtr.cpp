/**
 * @file
 */
#include "libbirch/LabelPtr.hpp"

#include "libbirch/Label.hpp"

libbirch::LabelPtr::LabelPtr(Label* ptr) :
    ptr(ptr) {
  if (ptr && ptr != root_label) {
    ptr->incShared();
  }
}

libbirch::LabelPtr::LabelPtr(const LabelPtr& o) {
  auto ptr = o.ptr.load();
  if (ptr && ptr != root_label) {
    ptr->incShared();
  }
  this->ptr.set(ptr);
}

libbirch::LabelPtr::LabelPtr(LabelPtr&& o) {
  ptr.set(o.ptr.exchange(nullptr));
}

libbirch::LabelPtr::~LabelPtr() {
  release();
}

void libbirch::LabelPtr::bitwiseFix(Label* label) {
  if (label && label != root_label) {
    label->incShared();
  }
  ptr.set(label);
}

libbirch::LabelPtr& libbirch::LabelPtr::operator=(const LabelPtr& o) {
  replace(o.get());
  return *this;
}

libbirch::LabelPtr& libbirch::LabelPtr::operator=(LabelPtr&& o) {
  auto ptr = o.ptr.exchange(nullptr);
  auto old = this->ptr.exchange(ptr);
  if (old && old != root_label) {
    old->decShared();
  }
  return *this;
}

bool libbirch::LabelPtr::query() const {
  return ptr.load() != nullptr;
}

libbirch::Label* libbirch::LabelPtr::get() const {
  return ptr.load();
}

void libbirch::LabelPtr::replace(Label* ptr) {
  if (ptr && ptr != root_label) {
    ptr->incShared();
  }
  auto old = this->ptr.exchange(ptr);
  if (old && old != root_label) {
    old->decShared();
  }
}

void libbirch::LabelPtr::release() {
  auto old = ptr.exchange(nullptr);
  if (old && old != root_label) {
    old->decShared();
  }
}

libbirch::Label& libbirch::LabelPtr::operator*() const {
  return *get();
}

libbirch::Label* libbirch::LabelPtr::operator->() const {
  return get();
}

void libbirch::LabelPtr::mark() {
  /* c.f. Shared::mark(); because we don't keep a shared reference to the root
   * label, it is not necessary to recurse into it */
  auto o = ptr.load();
  if (o && o != root_label) {
    o->breakShared();
    o->mark();
  }
}

void libbirch::LabelPtr::scan(const bool reachable) {
  /* c.f. Shared::mark(); because we don't keep a shared reference to the root
   * label, it is not necessary to recurse into it */
  auto o = ptr.load();
  if (o && o != root_label) {
    if (reachable) {
      o->incShared();
    }
    o->scan(reachable);
  }
}

void libbirch::LabelPtr::collect() {
  /* c.f. Shared::mark(); because we don't keep a shared reference to the root
   * label, it is not necessary to recurse into it */
  auto o = ptr.exchange(nullptr);
  if (o && o != root_label) {
    o->collect();
  }
}
