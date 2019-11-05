/*
 * Test beta-binomial cdf evaluations.
 */
program test_cdf_beta_binomial() {
  auto α <- simulate_uniform(1.0, 10.0);
  auto β <- simulate_uniform(1.0, 10.0);
  auto n <- simulate_uniform_int(1, 1000);
  
  ρ:Random<Real>;
  Beta(α, β).assume(ρ);
  auto q <- Binomial(n, ρ);
  test_cdf(q);
}
