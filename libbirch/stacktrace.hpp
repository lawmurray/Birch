/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/Allocator.hpp"

/**
 * @def libbirch_function_
 *
 * Push a new frame onto the stack trace.
 */
#ifndef NDEBUG
#define libbirch_function_(func, file, n) libbirch::StackFunction function_(func, file, n)
#else
#define libbirch_function_(func, file, n)
#endif

/**
 * @def libbirch_line_
 *
 * Update the line number of the top frame of the stack trace.
 */
#ifndef NDEBUG
#define libbirch_line_(n) libbirch::line(n)
#else
#define libbirch_line_(n)
#endif

namespace libbirch {
/**
 * Stack frame.
 */
struct StackFrame {
  const char* func;
  const char* file;
  int line;
};

/**
 * Temporary type for pushing functions onto the stack trace.
 */
struct StackFunction {
  StackFunction(const char* func, const char* file, const int line);
  ~StackFunction();
};

/**
 * Update the line number of the top frame of the stack trace.
 */
void line(const unsigned n);

/**
 * Print stack trace and abort.
 */
void abort();

/**
 * Print stack trace and abort with message.
 *
 * @param msg Message.
 * @param skip Number of frames on the top of the call stack to omit from the
 * stack trace.
 */
void abort(const std::string& msg, const unsigned skip = 0);
}
