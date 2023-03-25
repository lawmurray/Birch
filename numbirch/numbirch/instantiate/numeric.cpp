/**
 * @file
 */
#include "numbirch/common/numeric.inl"
#ifdef BACKEND_CUDA
#include "numbirch/cuda/numeric.inl"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/numeric.inl"
#endif

#define BINARY_MATRIX(f) \
    template Array<real,2> f<Array<real,0>,int>(const Array<real,2>&, const Array<real,0>&); \
    template Array<real,2> f<real,int>(const Array<real,2>&, const real&);

namespace numbirch {
BINARY_MATRIX(cholsolve)
BINARY_MATRIX(triinnersolve)
BINARY_MATRIX(trisolve)
}
