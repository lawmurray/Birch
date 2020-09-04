/*
 * Test Poisson cdf evaluations.
 */
program test_cdf_poisson(N:Integer <- 10000) {
  auto λ <- simulate_uniform(0.1, 100.0);
  auto q <- Poisson(λ);
  test_cdf(q);
}
