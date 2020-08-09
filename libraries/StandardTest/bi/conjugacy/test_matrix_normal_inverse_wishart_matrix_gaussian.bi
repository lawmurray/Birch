/*
 * Test matrix normal-inverse-Wishart-Gaussian conjugacy.
 */
program test_matrix_normal_inverse_wishart_matrix_gaussian(
    N:Integer <- 10000) {
  m:TestMatrixNormalInverseWishartMatrixGaussian;
  test_conjugacy(m, N, m.size());
}
