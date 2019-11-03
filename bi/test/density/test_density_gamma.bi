/*
 * Test gamma density evaluations.
 */
program test_density_gamma(N:Integer <- 10000) {
  auto k <- simulate_uniform(0.1, 10.0);
  auto θ <- simulate_uniform(0.0, 10.0);
  auto q <- Gamma(k, θ);  
  test_density(q, N);
}
