/*
 * Test bernoulli pdf evaluations.
 */
program test_pdf_bernoulli(N:Integer <- 10000) {
  auto ρ <- simulate_uniform(0.0, 1.0);
  auto q <- Bernoulli(ρ);
  test_pdf(q, N);
}
