/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"
#include "libbirch/Allocator.hpp"

#include <vector>

/**
 * @def bi_function
 *
 * Push a new frame onto the stack trace.
 */
#ifndef NDEBUG
#define bi_function(func, file, n) StackFunction function(func, file, n)
#else
#define bi_function(func, file, n)
#endif

/**
 * @def bi_line
 *
 * Update the line number of the top frame on the stack trace.
 */
#ifndef NDEBUG
#define bi_line(n) stacktrace.back().line = n
#else
#define bi_line(n)
#endif

namespace bi {
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
 * Stack trace.
 */
extern std::vector<StackFrame,Allocator<StackFrame>> stacktrace;
#pragma omp threadprivate(stacktrace)

/**
 * Print stack trace and abort.
 */
void abort();

/**
 * Print stack trace and abort with message.
 */
void abort(const std::string& msg);
}
