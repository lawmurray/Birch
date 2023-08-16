/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/transform.inl"
#include "numbirch/cuda/random.inl"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/transform.inl"
#include "numbirch/eigen/random.inl"
#endif
#include "numbirch/common/transform.inl"
#include "numbirch/common/random.inl"
#include "numbirch/instantiate/instantiate.hpp"

namespace numbirch {

NUMBIRCH_KEEP static void instantiate() {
  /* unary functions */
  std::visit([]<class T>(T x) {
    simulate_bernoulli(x);
    simulate_chi_squared(x);
    simulate_dirichlet(x);
    simulate_exponential(x);
    simulate_poisson(x);
  }, numeric_variant());

  /* binary functions */
  std::visit([]<class T, class U>(T x, U y) {
    /* exclude incompatible combinations, implicit_t is void for such */
    if constexpr (!std::is_same_v<implicit_t<T,U>,void>) {
      simulate_beta(x, y);
      simulate_binomial(x, y);
      simulate_gamma(x, y);
      simulate_gaussian(x, y);
      simulate_negative_binomial(x, y);
      simulate_weibull(x, y);
      simulate_uniform(x, y);
      simulate_uniform_int(x, y);
    }
  }, numeric_variant(), numeric_variant());

  /* Wishart function */
  std::visit([]<class T>(T x) {
    simulate_wishart(x, 0);
  }, scalar_variant());
}

}
