/*
 * Test scaled-gamma-exponential cdf evaluations.
 */
program test_cdf_scaled_gamma_exponential(N:Integer <- 10000) {
  m:TestScaledGammaExponential;
  m.play();
  test_cdf(m.marginal(), N);
}
