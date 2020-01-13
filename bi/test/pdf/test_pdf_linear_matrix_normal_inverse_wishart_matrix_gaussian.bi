/*
 * Test linear matrix normal-inverse-Wishart Gaussian pdf evaluations.
 */
program test_pdf_linear_matrix_normal_inverse_wishart_matrix_gaussian(
    N:Integer <- 10000, B:Integer <- 500, S:Integer <- 30) {
  m:TestLinearMatrixNormalInverseWishartMatrixGaussian;
  playDelay.handle(m.simulate());
  test_pdf(m.marginal(), 5, 2, N, B, S);
}
