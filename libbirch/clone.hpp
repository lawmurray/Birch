/**
 * @file
 */
#pragma once

#include "libbirch/Memo.hpp"

namespace libbirch {
/**
 * Is a clone currently underway?
 */
extern bool cloneUnderway;
#pragma omp threadprivate(cloneUnderway)

/**
 * Is a finishing clone currently underway?
 */
extern bool finishUnderway;
#pragma omp threadprivate(finishUnderway)

/**
 * The memo object associated with new objects.
 */
extern Memo* currentContext;
#pragma omp threadprivate(currentContext)

}
