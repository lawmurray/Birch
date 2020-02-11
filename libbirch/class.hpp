/**
 * @file
 */
#pragma once

#include "libbirch/Counted.hpp"

/**
 * @def libbirch_member_start_
 *
 * Boilerplate macro to occur first in a member function or fiber. Sets the
 * `self_` and `context_` variables.
 */
#define libbirch_member_start_ \
  [[maybe_unused]] libbirch::Label* context_(this->getLabel()); \
  [[maybe_unused]] libbirch::Lazy<libbirch::InitPtr<this_type_>> self(context_, this);

/**
 * @def libbirch_member_start_
 *
 * Boilerplate macro to occur first in a global function, fiber, or operator.
 * Sets the `context_` variable.
 */
#define libbirch_global_start_ \
  [[maybe_unused]] libbirch::Label* context_ = nullptr;

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
