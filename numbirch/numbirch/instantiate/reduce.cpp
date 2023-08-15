/**
 * @file
 */
#include "numbirch/common/reduce.inl"
#ifdef BACKEND_CUDA
#include "numbirch/cuda/transform.inl"
#include "numbirch/cuda/reduce.inl"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/transform.inl"
#include "numbirch/eigen/reduce.inl"
#endif
#include "numbirch/common/transform.inl"
#include "numbirch/instantiate/instantiate.hpp"

namespace numbirch {

[[maybe_unused]] static void instantiate() {
  std::visit([]<class T>(T x) {
    count(x);
    cumsum(x);
    min(x);
    max(x);
    sum(x);

    Array<real,0> g;
    Array<value_t<T>,0> y;
    count_grad(g, x);
    cumsum_grad(real_t<T>(), x);
    min_grad(g, y, x);
    max_grad(g, y, x);
    sum_grad(g, x);
  }, numeric_variant());
}

}
