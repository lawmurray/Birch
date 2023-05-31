/**
 * @file
 */
#include "birch/handler.hpp"

namespace birch {
/**
 * Event handler for each thread.
 */
static thread_local Handler handler(nullptr);

/**
 * Initialize.
 */
void init() {
  #pragma omp parallel
  {
    handler = birch::Handler(true, false, false);
  }
}

/**
 * Terminate.
 */
void term() {
  #pragma omp parallel
  {
    handler.release();
  }
}

/**
 * Get the event handler.
 */
Handler& get_handler() {
  return birch::handler;
}

/**
 * Set the event handler.
 */
void set_handler(const Handler& handler) {
  birch::handler = handler;
}

/**
 * Swap the event handler with another.
 *
 * @param handler The new handler.
 *
 * @return The previous handler.
 */
Handler swap_handler(const Handler& handler) {
  auto& current = birch::handler;
  auto previous = handler;
  std::swap(current, previous);
  return previous;
}

}
