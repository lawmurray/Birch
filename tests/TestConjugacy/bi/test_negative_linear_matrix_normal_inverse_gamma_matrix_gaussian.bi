/*
 * Test matrix linear normal-inverse-gamma-Gaussian conjugacy.
 */
program test_negative_linear_matrix_normal_inverse_gamma_matrix_gaussian(
    N:Integer <- 10000) {
  m:TestNegativeLinearMatrixNormalInverseGammaMatrixGaussian;
  test_conjugacy(m, N, m.size());
}
