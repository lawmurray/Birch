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
   * Depth.
   */
  static int depth() {
    return depth_impl<Sequence<Type>>::value;
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
}
