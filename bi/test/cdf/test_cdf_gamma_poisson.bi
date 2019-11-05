/*
 * Test gamma-Poisson cdf evaluations.
 */
program test_cdf_gamma_poisson() {
  auto k <- simulate_uniform_int(1, 10);
  auto θ <- simulate_uniform(0.0, 10.0);

  λ:Random<Real>;
  Gamma(k, θ).assume(λ);
  auto q <- Poisson(λ);
  test_cdf(q);
}
