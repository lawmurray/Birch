/**
 * @file
 */
#pragma once

#include <cassert>
#include <sstream>

/**
 * @def libbirch_assert_
 *
 * If debugging is enabled, check an assertion and abort on fail.
 */
#ifndef NDEBUG
#define libbirch_assert_(cond) if (!(cond)) libbirch::abort()
#else
#define libbirch_assert_(cond)
#endif

/**
 * @def libbirch_assert_msg_
 *
 * If debugging is enabled, check an assertion and abort with a message on
 * fail.
 */
#ifndef NDEBUG
#define libbirch_assert_msg_(cond, msg) \
  if (!(cond)) { \
    std::stringstream buf_; \
    buf_ << msg; \
    libbirch::abort(buf_.str()); \
  }
#else
#define libbirch_assert_msg_(cond, msg)
#endif

/**
 * @def libbirch_error_
 *
 * Check a condition and abort on fail.
 */
#define libbirch_error_(cond) if (!(cond)) libbirch::abort()

/**
 * @def libbirch_error_msg_
 *
 * Check a condition and abort with a message on fail.
 */
#define libbirch_error_msg_(cond, msg) \
  if (!(cond)) { \
    std::stringstream buf_; \
    buf_ << msg; \
    libbirch::abort(buf_.str()); \
  }
