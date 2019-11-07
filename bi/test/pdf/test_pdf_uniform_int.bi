/*
 * Test uniform integer pmf.
 */
program test_pdf_uniform_int(D:Integer <- 10, N:Integer <- 10000) {
  auto l <- simulate_uniform_int(-10, 10);
  auto u <- simulate_uniform_int(l, l + 20);
  auto π <- Uniform(l, u);
  test_pdf(π, N);
}
