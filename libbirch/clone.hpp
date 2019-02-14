/**
 * @file
 */
#pragma once

#include "libbirch/Memo.hpp"

namespace bi {
/**
 * Is a clone currently underway?
 */
extern bool cloneUnderway;
#pragma omp threadprivate(cloneUnderway)

/**
 * The memo object associated with new objects.
 */
extern Memo* currentContext;
#pragma omp threadprivate(currentContext)

}
