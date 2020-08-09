/*
 * Test normal-inverse-gamma-Gaussian conjugacy.
 */
program test_normal_inverse_gamma_gaussian(N:Integer <- 10000) {
  m:TestNormalInverseGammaGaussian;
  test_conjugacy(m, N, 3);
}
