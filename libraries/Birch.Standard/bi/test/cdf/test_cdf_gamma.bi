/*
 * Test gamma cdf evaluations.
 */
program test_cdf_gamma(N:Integer <- 10000) {
  auto k <- simulate_uniform(1.0, 10.0);
  auto θ <- simulate_uniform(0.0, 10.0);
  auto q <- Gamma(k, θ);  
  test_cdf(q, N);
}
