/**
 * @file
 */
#pragma once

#include "birch/basic.hpp"
#include "birch/utility.hpp"

namespace birch {
/**
 * Empty type, for unused arguments.
 */
struct Empty {
  //
};

template<>
struct is_empty<Empty> {
  static constexpr bool value = true;
};

template<>
struct tag_s<Empty> {
  using type = Empty;
};

template<>
struct peg_s<Empty> {
  using type = Empty;
};

}
