/**
 * @file
 */
#pragma once

#include "bi/data/View.hpp"

namespace bi {
/**
 * Iterator over an array.
 *
 * @ingroup library
 *
 * @tparam Value Value type.
 * @tparam Frame Frame type.
 * @tparam View View type.
 */
template<class Value, class Frame = EmptyFrame, class View = EmptyView>
class Iterator {
public:
  /**
   * Constructor.
   *
   * @param value Value.
   * @param frame Frame.
   * @param view View.
   */
  Iterator(const Value& value, const Frame& frame, const View& view, const int_t serial = 0) :
      value(value),
      frame(frame),
      view(view),
      serial(serial) {
    update();
  }

  /*
   * This needs to be a shallow copy, but semantics for Value are that
   * assignment does a deep copy, which is problematic. Delete to prevent
   * its use, as it may lead to difficult to debug errors.
   */
  Iterator<Value,Frame,View>& operator=(const Iterator<Value,Frame,View>& o) = delete;

  auto operator*();

  //Value* operator->() {
  //  return &result;
  //}

  bool operator==(const Iterator<Value,Frame,View>& o) const {
    return serial == o.serial;
  }

  bool operator!=(const Iterator<Value,Frame,View>& o) const {
    return !(*this == o);
  }

  Iterator<Value,Frame,View>& operator+=(const int_t i) {
    serial += i*view.length;
    update();
    return *this;
  }

  Iterator<Value,Frame,View>& operator+(const int_t i) {
    auto result = *this;
    result += i;
    return result;
  }

  Iterator<Value,Frame,View>& operator-=(const int_t i) {
    serial -= i*view.length;
    update();
    return *this;
  }

  Iterator<Value,Frame,View>& operator-(const int_t i) {
    auto result = *this;
    result -= i;
    return result;
  }

  Iterator<Value,Frame,View>& operator++() {
    serial += view.length;
    update();
    return *this;
  }

  Iterator<Value,Frame,View> operator++(int) {
    auto result = *this;
    ++*this;
    return result;
  }

  Iterator<Value,Frame,View>& operator--() {
    serial -= view.length;
    update();
    return *this;
  }

  Iterator<Value,Frame,View> operator--(int) {
    auto result = *this;
    --*this;
    return result;
  }

protected:
  /**
   * Update member attributes after serial index is changed.
   */
  void update() {
    frame.coord(serial, view);
  }

  /**
   * Value.
   */
  Value value;

  /**
   * Frame.
   */
  Frame frame;

  /**
   * View.
   */
  View view;

  /**
   * Serialised offset of the view.
   */
  int_t serial;
};
}

#include "bi/data/Array.hpp"

template<class Value, class Frame, class View>
auto bi::Iterator<Value,Frame,View>::operator*() {
  Array<Value,Frame> array(value, frame);
  return array(view);
}
