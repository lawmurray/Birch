/*
 * Test multivariate inverse gamma evaluations.
 */
program test_pdf_multivariate_inverse_gamma(D:Integer <- 4,
    N:Integer <- 10000, B:Integer <- 500, S:Integer <- 20) {
  auto α <- simulate_uniform(2.0, 10.0);
  β:Real[D];
  for auto i in 1..D {
    β[i] <- simulate_uniform(0.0, 10.0);
  }
  auto π <- InverseGamma(α, β);
  test_pdf(π, D, N, B, S);
}
