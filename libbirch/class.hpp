/**
 * @file
 */
#pragma once

/**
 * @def libbirch_swap_context_
 *
 * When lazy deep clone is in use, swaps into the context of this object.
 */
#define libbirch_swap_context_ [[maybe_unused]] auto context_ = libbirch::Any::getLabel();

/**
 * @def libbirch_declare_self_
 *
 * Declare `self` within a member function.
 */
#define libbirch_declare_self_ libbirch::Lazy<libbirch::InitPtr<this_type_>> self(this->getLabel(), this);

#include "libbirch/Any.hpp"

namespace bi {
  namespace type {
/**
 * Super type of another. This is specialized for all classes that are
 * derived from Any to indicate their super type without having to
 * instantiate that type.
 */
template<class T>
struct super_type {
  using type = libbirch::Any;
};
  }
}
