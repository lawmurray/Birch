/**
 * @file
 */
#include "numbirch/random.hpp"

#include "numbirch/eigen/stl.hpp"

#if HAVE_OMP_H
#include <omp.h>
#endif

namespace numbirch {

void seed(const int s) {
  #pragma omp parallel
  {
    #if HAVE_OMP_H
    int n = omp_get_thread_num();
    int N = omp_get_max_threads();
    #else
    int n = 0;
    int N = 1;
    #endif

    /* fine to use the same seed here, as these are different algorithms
     * and/or parameterizations */
    rng32.seed(s*N + n);
    rng64.seed(s*N + n);
  }
}

}
