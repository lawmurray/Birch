/*
 * Test beta cdf evaluations.
 */
program test_cdf_beta(N:Integer <- 10000) {
  auto α <- simulate_uniform(1.0, 10.0);
  auto β <- simulate_uniform(1.0, 10.0);
  auto q <- Beta(α, β);  
  test_cdf(q, N);
}
