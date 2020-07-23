/*
 * Test gamma-Poisson conjugacy.
 */
program test_gamma_poisson(N:Integer <- 10000) {
  m:TestGammaPoisson;
  test_conjugacy(m, N, 2);
}
