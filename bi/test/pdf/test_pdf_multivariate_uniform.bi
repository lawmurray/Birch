/*
 * Test multivariate uniform pdf evaluations.
 */
program test_pdf_multivariate_uniform(D:Integer <- 4, N:Integer <- 10000,
    B:Integer <- 10000, S:Integer <- 10) {
  l:Real[D];
  u:Real[D];

  for auto i in 1..D {
    l[i] <- simulate_uniform(-10.0, 10.0);
    u[i] <- simulate_uniform(l[i], l[i] + 20.0);
  }

  auto π <- Uniform(l, u);
  test_pdf(π, D, N, B, S);
}
