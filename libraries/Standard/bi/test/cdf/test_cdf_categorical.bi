/*
 * Test categorical cdf evaluations.
 */
program test_cdf_categorical(N:Integer <- 10000) {
  auto n <- simulate_uniform_int(10, 50);
  auto ρ <- simulate_independent_uniform(vector(0.0, n), vector(1.0, n));
  ρ <- ρ/sum(ρ);
  auto q <- Categorical(ρ);
  test_cdf(q);
}
