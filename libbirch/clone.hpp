/**
 * @file
 */
#pragma once

#include "libbirch/Memo.hpp"

namespace libbirch {
/**
 * Is a clone currently underway?
 */
#pragma omp declare target
/*thread_local*/ extern bool cloneUnderway;
#pragma omp end declare target

/**
 * The memo object associated with new objects.
 */
#pragma omp declare target
/*thread_local*/ extern Memo* currentContext;
#pragma omp end declare target

}
