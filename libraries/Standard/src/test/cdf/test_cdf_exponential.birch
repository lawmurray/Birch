/*
 * Test exponential cdf evaluations.
 */
program test_cdf_exponential(N:Integer <- 10000) {
  auto λ <- simulate_uniform(1.0, 10.0);
  auto q <- Exponential(λ);  
  test_cdf(q, N);
}
