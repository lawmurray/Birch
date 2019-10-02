/**
 * @file
 */
#include "libbirch/stacktrace.hpp"

struct stack_frame {
  const char* func;
  const char* file;
  int line;
};
using stack_trace = std::vector<stack_frame,libbirch::Allocator<stack_frame>>;

stack_trace& currentStackTrace() {
  static std::vector<stack_trace,libbirch::Allocator<stack_trace>> stacktraces(
      omp_get_max_threads());
  return stacktraces[omp_get_thread_num()];
}

libbirch::StackFunction::StackFunction(const char* func, const char* file,
    const int line) {
  currentStackTrace().push_back({ func, file, line });
}

libbirch::StackFunction::~StackFunction() {
  currentStackTrace().pop_back();
}

void libbirch::line(const unsigned n) {
  currentStackTrace().back().line = n;
}

void libbirch::abort() {
  abort("assertion failed");
}

void libbirch::abort(const std::string& msg, const unsigned skip) {
  printf("error: %s\n", msg.c_str());
  #ifndef NDEBUG
  printf("stack trace:\n");
  auto stacktrace = currentStackTrace();
  auto i = 0;
  for (auto iter = stacktrace.rbegin() + skip; (i < 20 + skip) &&
      iter != stacktrace.rend(); ++iter) {
    if (iter->file) {
      printf("    %-24s @ %s:%d\n", iter->func, iter->file, iter->line);
    } else {
      printf("    %-24s\n", iter->func);
    }
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
