/*
 * Test that an invalid Gaussian-Gaussian relationship is not determined.
 */
program test_invalid_gaussian_gaussian() {
  m:TestInvalidGaussianGaussian;
  handle(m.simulate());
  
  /* if y is instantiated now, x should also be instantiated; establishing
   * a Gaussian-Gaussian relationship when x appears in the variance of y
   * is invalid. */
  m.y.value();
  if !m.x.hasValue() {
    exit(1);
  }
}

class TestInvalidGaussianGaussian() < Model {
  x:Random<Real>;
  y:Random<Real>;
  
  fiber simulate() -> Event {
    x ~ Gaussian(0.0, 1.0);
    y ~ Gaussian(x, x*x);
  }
}
