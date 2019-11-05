/*
 * Test Weibull cdf evaluations.
 */
program test_cdf_weibull(N:Integer <- 10000) {
  auto k <- simulate_uniform(1.0, 10.0);
  auto λ <- simulate_uniform(0.1, 10.0);
  auto q <- Weibull(k, λ);  
  test_cdf(q, N);
}
