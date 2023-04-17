/**
 * @file
 */
#pragma once

namespace birch {
/**
 * Initialize.
 */
void init();

/**
 * Terminate.
 */
void term();

/**
 * Get the event handler.
 */
Handler& get_handler();

/**
 * Set the event handler.
 */
void set_handler(const Handler& handler);

/**
 * Swap the event handler with another.
 *
 * @param handler The new handler.
 *
 * @return The previous handler.
 */
Handler swap_handler(const Handler& handler);

}
