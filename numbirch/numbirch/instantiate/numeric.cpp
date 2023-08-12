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
    template Array<real,2> f<Array<real,0>>(const Array<real,2>&, const Array<real,0>&); \
    template Array<real,2> f<real>(const Array<real,2>&, const real&); \
    template Array<real,2> f##_grad1<Array<real,0>>(const Array<real,2>& g, \
        const Array<real,2>& B, const Array<real,2>& L, const Array<real,0>& y); \
    template Array<real,2> f##_grad1<real>(const Array<real,2>& g, \
        const Array<real,2>& B, const Array<real,2>& L, const real& y); \
    template Array<real,0> f##_grad2<Array<real,0>>(const Array<real,2>& g, \
        const Array<real,2>& B, const Array<real,2>& L, const Array<real,0>& y); \
    template Array<real,0> f##_grad2<real>(const Array<real,2>& g, \
        const Array<real,2>& B, const Array<real,2>& L, const real& y);

namespace numbirch {
BINARY_MATRIX(cholsolve)
BINARY_MATRIX(triinnersolve)
BINARY_MATRIX(trisolve)

template Array<bool,2> transpose<bool>(const Array<bool,2>&);
template Array<int,2> transpose<int>(const Array<int,2>&);
template Array<real,2> transpose<real>(const Array<real,2>&);

template Array<real,2> transpose_grad<bool>(const Array<real,2>&,const Array<bool,2>&);
template Array<real,2> transpose_grad<int>(const Array<real,2>&, const Array<int,2>&);
template Array<real,2> transpose_grad<real>(const Array<real,2>&, const Array<real,2>&);

}
