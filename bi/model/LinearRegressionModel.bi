class LinearRegressionModel < Model {
  /**
   * Explanatory variables.
   */
  X:Real[_,_];

  /**
   * Regression coefficients.
   */
  β:Random<Real[_]>;

  /**
   * Observation variance.
   */
  σ2:Random<Real>;

  /**
   * Observations.
   */
  y:Random<Real[_]>;
  
  fiber simulate() -> Real {
    N:Integer <- rows(X);
    P:Integer <- columns(X);
    if (N > 0 && P > 0) {
      σ2 ~ InverseGamma(3.0, 0.4);
      β ~ Gaussian(vector(0.0, P), identity(P)*σ2);
      y ~ Gaussian(X*β, σ2);
    }
  }
  
  function input(reader:Reader) {
    X <- reader.getRealMatrix("X")!;
    y <- reader.getRealVector("y")!;
  }
  
  function output(writer:Writer) {
    writer.setRealVector("β", β);
    writer.setReal("σ2", σ2);
  }
}
