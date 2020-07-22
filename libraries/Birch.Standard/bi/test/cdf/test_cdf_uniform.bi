/*
 * Test uniform cdf evaluations.
 */
program test_cdf_uniform(N:Integer <- 10000) {
  auto l <- simulate_uniform(-10.0, 10.0);
  auto u <- simulate_uniform(l, l + 20.0);
  auto q <- Uniform(l, u);  
  test_cdf(q, N);
}
