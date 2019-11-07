/*
 * Test matrix normal-inverse-Wishart Gaussian pdf evaluations.
 */
program test_pdf_matrix_normal_inverse_wishart_matrix_gaussian(
    N:Integer <- 10000, B:Integer <- 50000, S:Integer <- 20) {
  m:TestMatrixNormalInverseWishartMatrixGaussian;
  m.play();
  test_pdf(m.marginal(), 5, 2, N, B, S);
}
