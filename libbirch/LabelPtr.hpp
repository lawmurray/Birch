/**
 * @file
 */
#pragma once

#include "libbirch/Label.hpp"

namespace libbirch {
/**
 * Pointer to a Label object. When the referent is the current context, this
 * acts as a raw pointer, otherwise as as a shared pointer.
 *
 * @ingroup libbirch
 */
class LabelPtr {
public:
  /**
   * Default constructor.
   */
  LabelPtr() : label(0), cross(false) {
    //
  }

  /**
   * Value constructor.
   */
  explicit LabelPtr(Label* label) {
    set(label);
  }

  /**
   * Copy constructor.
   */
  LabelPtr(const LabelPtr& o) {
    set(o.get());
  }

  /**
   * Destructor.
   */
  ~LabelPtr() {
    release();
  }

  /**
   * Copy assignment.
   */
  LabelPtr& operator=(const LabelPtr& o) {
    replace(o.get());
    return *this;
  }

  /**
   * Get the raw pointer.
   */
  Label* get() const {
    return reinterpret_cast<Label*>(label);
  }

  /**
   * Replace.
   */
  void replace(Label* label) {
    auto old = get();
    if (label != old) {
      auto oldCross = isCross();
      set(label);
      if (old && oldCross) {
        old->decShared();
      }
    }
  }

  /**
   * Release the context.
   */
  void release() {
    if (isCross()) {
      get()->decShared();
    }
    label = 0;
    cross = false;
  }

  /**
   * Is this pointer crossed? A crossed pointer is to a context different to
   * that of the context in which it was created (e.g. the context of the
   * object to which it belongs).
   */
  bool isCross() const {
    return cross;
  }

  /**
   * Dereference.
   */
  Label& operator*() const {
    return *get();
  }

  /**
   * Member access.
   */
  Label* operator->() const {
    return get();
  }

  /**
   * Equal comparison.
   */
  bool operator==(const LabelPtr& o) const {
    return get() == o.get();
  }

  /**
   * Not equal comparison.
   */
  bool operator!=(const LabelPtr& o) const {
    return get() != o.get();
  }

  /**
   * Is the pointer not null?
   */
  operator bool() const {
    return label != 0;
  }

private:
  /**
   * Set.
   */
  void set(Label* label) {
    this->label = reinterpret_cast<intptr_t>(label);
    cross = label/* && label != currentContext*/;
    if (cross) {
      label->incShared();
    }
  }

  /**
   * Raw pointer.
   */
  intptr_t label:63;

  /**
   * Is this a cross pointer?
   */
  bool cross:1;
};
}
