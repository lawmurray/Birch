/**
 * @file
 */
#pragma once

/**
 * @def libbirch_swap_context_
 *
 * When lazy deep clone is in use, swaps into the context of this object.
 */
#if ENABLE_LAZY_DEEP_CLONE
#define libbirch_swap_context_ [[maybe_unused]] auto label_ = libbirch::Any::getLabel();
#else
#define libbirch_swap_context_
#endif

/**
 * @def libbirch_declare_self_
 *
 * Declare `self` within a member function.
 */
#define libbirch_declare_self_ libbirch::Init<this_type_> self(label_, this);

/**
 * @def libbirch_declare_local_
 *
 * Declare `local` within a member fiber.
 */
#define libbirch_declare_local_ libbirch::Init<class_type_> local(label_, this);
