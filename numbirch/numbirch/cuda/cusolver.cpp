/**
 * @file
 */
#include "numbirch/cuda/cusolver.hpp"
#include "numbirch/cuda/cuda.hpp"
#include "numbirch/cuda/cublas.hpp"

#if HAVE_OMP_H
#include <omp.h>
#endif

namespace numbirch {

thread_local cusolverDnHandle_t cusolverDnHandle;
thread_local cusolverDnParams_t cusolverDnParams;

void cusolver_init() {
  #pragma omp parallel num_threads(omp_get_max_threads())
  {
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverDnHandle));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverDnHandle, stream));
    CUSOLVER_CHECK(cusolverDnCreateParams(&cusolverDnParams));
  }
}

void cusolver_term() {
  #pragma omp parallel num_threads(omp_get_max_threads())
  {
    CUSOLVER_CHECK(cusolverDnDestroyParams(cusolverDnParams));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverDnHandle));
  }
}

}
