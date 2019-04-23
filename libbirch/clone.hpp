/**
 * @file
 */
#pragma once

#include "libbirch/Memo.hpp"

namespace libbirch {
/**
 * Is a clone currently underway?
 */
thread_local extern bool cloneUnderway;

/**
 * The memo object associated with new objects.
 */
thread_local extern Memo* currentContext;

}
