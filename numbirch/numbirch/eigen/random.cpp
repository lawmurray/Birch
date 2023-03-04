/**
 * @file
 */
#include "numbirch/common/random.hpp"
#include "numbirch/eigen/eigen.hpp"
#include "numbirch/array.hpp"

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

Array<real,1> convolve(const Array<real,1>& p, const Array<real,1>& q) {
  assert(stride(p) == 1);
  int m = length(p);
  int n = length(q);
  Array<real,1> r(make_shape(m + n - 1));

  auto L = make_eigen(sliced(p), m, n, -1).
      template triangularView<Eigen::Lower>();
  auto U = make_eigen(sliced(p) + m, n - 1, n, - 1).
      template triangularView<Eigen::StrictlyUpper>();
  auto q1 = make_eigen(q);
  auto r1 = make_eigen(r);

  r1.head(m).noalias() = L*q1;
  r1.tail(n - 1).noalias() = U*q1;
  return r;
}

Array<real,1> convolve_grad1(const Array<real,1>& g, const Array<real,1>& r,
    const Array<real,1>& p, const Array<real,1>& q) {
  assert(stride(g) == 1);
  int m = length(p);
  int n = length(q);
  Array<real,1> gp(make_shape(m));

  auto gp1 = make_eigen(gp);
  auto B = make_eigen(sliced(g), m, n, 0);
  auto q1 = make_eigen(q);

  gp1.noalias() = B*q1;
  return gp;
}

Array<real,1> convolve_grad2(const Array<real,1>& g, const Array<real,1>& r,
    const Array<real,1>& p, const Array<real,1>& q) {
  assert(stride(p) == 1);
  int m = length(p);
  int n = length(q);
  Array<real,1> gq(make_shape(n));

  auto L = make_eigen(sliced(p), m, n, -1).
      template triangularView<Eigen::Lower>();
  auto U = make_eigen(sliced(p) + m, n - 1, n, - 1).
      template triangularView<Eigen::StrictlyUpper>();
  auto gq1 = make_eigen(gq);
  auto g1 = make_eigen(g);

  gq1.noalias() = L.transpose()*g1.head(m) + U.transpose()*g1.tail(n - 1);
  return gq;
}

}
