/**
 * @file
 */
#include "libbirch/stacktrace.hpp"

#include "libbirch/thread.hpp"

/**
 * Stack frame.
 */
struct stack_frame {
  const char* func;
  const char* file;
  int line;
};

static thread_local std::vector<stack_frame> stack_trace;

libbirch::StackFunction::StackFunction(const char* func, const char* file,
    const int line) {
  stack_trace.push_back({ func, file, line });
}

libbirch::StackFunction::~StackFunction() {
  stack_trace.pop_back();
}

void libbirch::line(const int n) {
  stack_trace.back().line = n;
}

void libbirch::abort() {
  abort("assertion failed");
}

void libbirch::abort(const std::string& msg, const int skip) {
  printf("error: %s\n", msg.c_str());
  #ifndef NDEBUG
  printf("stack trace:\n");
  auto& trace = stack_trace;
  int i = 0;
  for (auto iter = trace.rbegin() + skip; (i < 20 + skip) &&
      iter != trace.rend(); ++iter) {
    if (iter->file) {
      printf("    %-24s @ %s:%d\n", iter->func, iter->file, iter->line);
    } else {
      printf("    %-24s\n", iter->func);
    }
    ++i;
  }
  if (i < (int)trace.size() - skip) {
    int rem = (int)trace.size() - skip - i;
    printf("  + %d more\n", rem);
  }
  assert(false);
  #else
  std::exit(1);
  #endif
}
