/**
 * @file
 */
#pragma once

#include "bi/lib/global.hpp"

#include <initializer_list>

namespace bi {


/**
 * Sequence.
 *
 * @ingroup library
 *
 * @tparam Type Value type.
 */
template<class Type>
class Sequence {
public:
  /**
   * Constructor.
   */
  Sequence(const std::initializer_list<Type>& values) : values(values) {
    //
  }

  /**
   * Width.
   */
  size_t width() const {
    return values.size();
  }

  /**
   * Depth.
   */
  static int depth() {
    return depth_impl<Sequence<Type>>::value;
  }

  /**
   * Does this sequence conform in size to the frame of an array?
   */
  template<class Frame>
  bool conforms(const Frame& frame) const {
    if (frame.count() != depth()) {
      return false;
    } else {
      size_t lengths[depth()];
      frame.lengths(lengths);
      return conforms(lengths);
    }
  }

  /*
   * Iterators.
   */
  auto begin() {
    return values.begin();
  }
  auto begin() const {
    return values.begin();
  }
  auto end() {
    return values.end();
  }
  auto end() const {
    return values.end();
  }

private:
  /**
   * Values.
   */
  std::initializer_list<Type> values;

  /**
   * Implementation of conforms().
   */
  bool conforms(const size_t lengths[]) const {
    if (lengths[0] != width()) {
      return false;
    } else if (depth() > 1) {
      for (auto o: values) {
        if (!o.conforms(lengths + 1)) {
          return false;
        }
      }
    }
    return true;
  }

  /**
   * Depth of a sequence.
   */
  template<class Type1>
  struct depth_impl {
    static const int value = 0;
  };
  template<class Type1>
  struct depth_impl<Sequence<Type1>> {
    static const int value = 1 + depth_impl<Type1>::value;
  };
};

template<class Type, class Iterator>
void copy(const Sequence<Type>& from, Iterator& to) {
  for (auto o: from) {
    copy(o, to);
  }
}

template<class Type, class Iterator>
void copy(const Type& from, Iterator& to) {
  *to = from;
  ++to;
}

}
