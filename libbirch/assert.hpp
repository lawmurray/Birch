/**
 * @file
 */
#pragma once

#include <cassert>
#include <sstream>

/**
 * @def bi_assert
 *
 * If debugging is enabled, check an assertion and abort on fail.
 */
#ifndef NDEBUG
#define bi_assert(cond) if (!(cond)) bi::abort()
#else
#define bi_assert(cond)
#endif

/**
 * @def bi_assert_msg
 *
 * If debugging is enabled, check an assertion and abort with a message on
 * fail.
 */
#ifndef NDEBUG
#define bi_assert_msg(cond, msg) \
  if (!(cond)) { \
    std::stringstream buf; \
    buf << msg; \
    bi::abort(buf.str()); \
  }
#else
#define bi_assert_msg(cond, msg)
#endif
