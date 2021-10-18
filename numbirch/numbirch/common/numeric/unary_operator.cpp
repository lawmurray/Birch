/**
 * @file
 * 
 * Explicit instantiations of numeric functions for the enabled backend.
 */
#include "numbirch/numeric/unary_operator.hpp"

#ifdef BACKEND_ONEAPI
#include "numbirch/oneapi/numeric/unary_operator.hpp"
#endif
#ifdef BACKEND_CUDA
#include "numbirch/cuda/numeric/unary_operator.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/numeric/unary_operator.hpp"
#endif

namespace numbirch {

void logical_not(const int m, const int n, const bool* A, const int ldA,
    bool* B, const int ldB);

template void neg(const int m, const int n, const double* A, const int ldA,
    double* B, const int ldB);
template void neg(const int m, const int n, const float* A, const int ldA,
    float* B, const int ldB);
template void neg(const int m, const int n, const int* A, const int ldA,
    int* B, const int ldB);

}
