/*
 * Test binomial cdf evaluations.
 */
program test_cdf_binomial(N:Integer <- 10000) {
  auto n <- simulate_uniform_int(0, 1000);
  auto ρ <- simulate_uniform(0.0, 1.0);
  auto q <- Binomial(n, ρ);
  test_cdf(q);
}
