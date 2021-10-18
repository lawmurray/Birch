/**
 * @file
 * 
 * Explicit instantiations of numeric functions for the enabled backend.
 */
#include "numbirch/numeric/other_function.hpp"

#ifdef BACKEND_ONEAPI
#include "numbirch/oneapi/numeric/other_function.hpp"
#endif
#ifdef BACKEND_CUDA
#include "numbirch/cuda/numeric/other_function.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/numeric/other_function.hpp"
#endif

namespace numbirch {

template void combine(const int m, const int n, const double a,
    const double* A, const int ldA, const double b, const double* B,
    const int ldB, const double c, const double* C, const int ldC,
    const double d, const double* D, const int ldD, double* E, const int ldE);
template void combine(const int m, const int n, const float a,
    const float* A, const int ldA, const float b, const float* B,
    const int ldB, const float c, const float* C, const int ldC,
    const float d, const float* D, const int ldD, float* E, const int ldE);

}
