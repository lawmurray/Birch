/*
 * Test geometric cdf evaluations.
 */
program test_cdf_geometric() {
  auto ρ <- simulate_uniform(0.0, 1.0);
  auto q <- Geometric(ρ);  
  test_cdf(q);
}
