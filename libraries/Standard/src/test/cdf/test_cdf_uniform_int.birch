/*
 * Test uniform integer cdf evaluations.
 */
program test_cdf_uniform_int(N:Integer <- 10000) {
  auto l <- simulate_uniform_int(-100, 100);
  auto u <- simulate_uniform_int(l, l + 200);
  auto q <- Uniform(l, u);  
  test_cdf(q);
}
