/**
 * @file
 */
#include "libbirch/stacktrace.hpp"

/**
 * Stack trace.
 */
thread_local static std::vector<libbirch::StackFrame,libbirch::Allocator<libbirch::StackFrame>> stacktrace;

libbirch::StackFunction::StackFunction(const char* func, const char* file,
    const int line) {
  stacktrace.push_back( { func, file, line });
}

libbirch::StackFunction::~StackFunction() {
  stacktrace.pop_back();
}

void libbirch::line(const unsigned n) {
  stacktrace.back().line = n;
}

void libbirch::abort() {
  abort("assertion failed");
}

void libbirch::abort(const std::string& msg, const unsigned skip) {
  printf("error: %s\n", msg.c_str());
  #ifndef NDEBUG
  printf("stack trace:\n");
  unsigned i = 0;
  for (auto iter = stacktrace.rbegin() + skip; (i < 20u + skip) &&
      iter != stacktrace.rend(); ++iter) {
    printf("    %-24s @ %s:%d\n", iter->func, iter->file, iter->line);
    ++i;
  }
  if (i < stacktrace.size() - skip) {
    int rem = stacktrace.size() - skip - i;
    printf("  + %d more\n", rem);
  }
  assert(false);
  #else
  std::exit(1);
  #endif
}
