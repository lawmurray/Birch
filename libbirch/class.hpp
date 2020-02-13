/**
 * @file
 */
#pragma once

#include "libbirch/Counted.hpp"

/**
 * @def libbirch_member_start_
 *
 * Boilerplate macro to occur first in a member function or fiber. Sets the
 * `self`.
 */
#define libbirch_member_start_ \
  [[maybe_unused]] libbirch::Lazy<libbirch::InitPtr<this_type_>> self(this->getLabel(), this);

namespace bi {
  namespace type {
/**
 * Super type of another. This is specialized for all classes that are
 * derived from Any to indicate their super type without having to
 * instantiate that type.
 */
template<class T>
struct super_type {
  //
};
  }
}
