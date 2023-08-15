/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/transform.inl"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/transform.inl"
#endif
#include "numbirch/common/transform.inl"
#include "numbirch/instantiate/instantiate.hpp"

namespace numbirch {

[[maybe_unused]] static void instantiate() {
  /* unary functions */
  std::visit([]<class T>(T x) {
    abs(x);
    acos(x);
    asin(x);
    atan(x);
    cast<real>(x);
    cast<int>(x);
    cast<bool>(x);
    ceil(x);
    cos(x);
    cosh(x);
    digamma(x);
    erf(x);
    exp(x);
    expm1(x);
    floor(x);
    isfinite(x);
    isinf(x);
    isnan(x);
    lfact(x);
    lgamma(x);
    log(x);
    log1p(x);
    logical_not(x);
    neg(x);
    pos(x);
    rectify(x);
    round(x);
    sin(x);
    sinh(x);
    sqrt(x);
    tan(x);
    tanh(x);

    real_t<T> g;
    abs_grad(g, x);
    acos_grad(g, x);
    asin_grad(g, x);
    atan_grad(g, x);
    ceil_grad(g, x);
    cos_grad(g, x);
    cosh_grad(g, x);
    exp_grad(g, x);
    expm1_grad(g, x);
    floor_grad(g, x);
    isfinite_grad(g, x);
    isinf_grad(g, x);
    isnan_grad(g, x);
    lfact_grad(g, x);
    lgamma_grad(g, x);
    log_grad(g, x);
    log1p_grad(g, x);
    logical_not_grad(g, x);
    neg_grad(g, x);
    pos_grad(g, x);
    rectify_grad(g, x);
    round_grad(g, x);
    sin_grad(g, x);
    sinh_grad(g, x);
    sqrt_grad(g, x);
    tan_grad(g, x);
    tanh_grad(g, x);
  }, numeric_variant());

  /* binary functions */
  std::visit([]<class T, class U>(T x, U y) {
    /* exclude incompatible combinations, implicit_t is void for such */
    if constexpr (!std::is_same_v<implicit_t<T,U>,void>) {
      add(x, y);
      copysign(x, y);
      div(x, y);
      digamma(x, y);
      equal(x, y);
      gamma_p(x, y);
      gamma_q(x, y);
      greater(x, y);
      greater_or_equal(x, y);
      hadamard(x, y);
      lbeta(x, y);
      lchoose(x, y);
      less(x, y);
      less_or_equal(x, y);
      lgamma(x, y);
      logical_and(x, y);
      logical_or(x, y);
      not_equal(x, y);
      pow(x, y);
      sub(x, y);

      real_t<T,U> g;
      add_grad1(g, x, y);
      add_grad2(g, x, y);
      copysign_grad1(g, x, y);
      copysign_grad2(g, x, y);
      div_grad1(g, x, y);
      div_grad2(g, x, y);
      equal_grad1(g, x, y);
      equal_grad2(g, x, y);
      greater_grad1(g, x, y);
      greater_grad2(g, x, y);
      greater_or_equal_grad1(g, x, y);
      greater_or_equal_grad2(g, x, y);
      hadamard_grad1(g, x, y);
      hadamard_grad2(g, x, y);
      lbeta_grad1(g, x, y);
      lbeta_grad2(g, x, y);
      lchoose_grad1(g, x, y);
      lchoose_grad2(g, x, y);
      less_grad1(g, x, y);
      less_grad2(g, x, y);
      less_or_equal_grad1(g, x, y);
      less_or_equal_grad2(g, x, y);
      lgamma_grad1(g, x, y);
      lgamma_grad2(g, x, y);
      logical_and_grad1(g, x, y);
      logical_and_grad2(g, x, y);
      logical_or_grad1(g, x, y);
      logical_or_grad2(g, x, y);
      not_equal_grad1(g, x, y);
      not_equal_grad2(g, x, y);
      pow_grad1(g, x, y);
      pow_grad2(g, x, y);
      sub_grad1(g, x, y);
      sub_grad2(g, x, y);
    }
  }, numeric_variant(), numeric_variant());

  /* ternary functions */
  std::visit([]<class T, class U, class V>(T x, U y, V z) {
    /* exclude incompatible combinations, implicit_t is void for such */
    if constexpr (!std::is_same_v<implicit_t<T,U,V>,void>) {
      ibeta(x, y, z);
      lz_conway_maxwell_poisson(x, y, z);
      where(x, y, z);

      real_t<T,U,real> g;
      lz_conway_maxwell_poisson_grad1(g, x, y, z);
      lz_conway_maxwell_poisson_grad2(g, x, y, z);
      lz_conway_maxwell_poisson_grad3(g, x, y, z);
      where_grad1(g, x, y, z);
      where_grad2(g, x, y, z);
      where_grad3(g, x, y, z);
    }
  }, numeric_variant(), numeric_variant(), numeric_variant());
}

}
