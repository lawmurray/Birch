/*
 * Test inverse-gamma cdf evaluations.
 */
program test_cdf_inverse_gamma(N:Integer <- 10000) {
  auto α <- simulate_uniform(2.0, 10.0);
  auto β <- simulate_uniform(0.1, 10.0);
  auto q <- InverseGamma(α, β);  
  test_cdf(q, N);
}
