/*
 * Test beta density evaluations.
 */
program test_density_beta(N:Integer <- 10000) {
  auto α <- simulate_uniform(1.0, 10.0);
  auto β <- simulate_uniform(1.0, 10.0);
  auto q <- Beta(α, β);  
  test_density(q, N);
}
