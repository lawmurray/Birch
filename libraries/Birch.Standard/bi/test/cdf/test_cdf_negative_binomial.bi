/*
 * Test negative binomial cdf evaluations.
 */
program test_cdf_negative_binomial() {
  auto k <- simulate_uniform_int(1, 20);
  auto ρ <- simulate_uniform(0.0, 1.0);
  auto q <- NegativeBinomial(k, ρ);
  test_cdf(q);
}
