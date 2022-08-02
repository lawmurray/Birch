/**
 * @file
 */
#include "numbirch/common/random.hpp"

#if HAVE_OMP_H
#include <omp.h>
#endif

namespace numbirch {

void seed(const int s) {
  #pragma omp parallel num_threads(omp_get_max_threads())
  {
    #if HAVE_OMP_H
    auto n = omp_get_thread_num();
    auto N = omp_get_max_threads();
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

void seed() {
  #pragma omp parallel num_threads(omp_get_max_threads())
  {
    std::random_device rd;
    rng32.seed(rd());
    rng64.seed(rd());
  }
}

}
