/*
 * Test Poisson density evaluations.
 */
program test_density_poisson() {
  auto λ <- simulate_uniform(0.1, 100.0);
  auto q <- Poisson(λ);
  test_density(q);
}
