/**
 * @file
 */
#pragma once

#include "libbirch/Context.hpp"

namespace libbirch {
/**
 * Is a clone currently underway in the thread?
 *
 * @note A thread-local global variable is used rather than a function
 * argument to facilitate use of copy constructors when cloning objects,
 * particularly important when using types from e.g. the STL where we can't
 * modify the copy constructor or add new constructors.
 */
thread_local extern bool cloneUnderway;

/**
 * The current context to which to assign newly created objects in the
 * thread.
 */
thread_local extern Context* currentContext;

}
