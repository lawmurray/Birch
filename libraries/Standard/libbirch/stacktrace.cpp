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

/**
 * Get the stack trace for the current thread.
 */
static auto& get_thread_stack_trace() {
  using stack_trace = std::vector<stack_frame,libbirch::Allocator<stack_frame>>;
  static std::vector<stack_trace,libbirch::Allocator<stack_trace>> stack_traces(
      libbirch::get_max_threads());
  return stack_traces[libbirch::get_thread_num()];
}

libbirch::StackFunction::StackFunction(const char* func, const char* file,
    const int line) {
  get_thread_stack_trace().push_back({ func, file, line });
}

libbirch::StackFunction::~StackFunction() {
  get_thread_stack_trace().pop_back();
}

void libbirch::line(const int n) {
  get_thread_stack_trace().back().line = n;
}

void libbirch::abort() {
  abort("assertion failed");
}

void libbirch::abort(const std::string& msg, const int skip) {
  printf("error: %s\n", msg.c_str());
  #ifndef NDEBUG
  printf("stack trace:\n");
  auto& trace = get_thread_stack_trace();
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
