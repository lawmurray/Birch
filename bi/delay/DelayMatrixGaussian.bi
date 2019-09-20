/*
 * Delayed matrix Gaussian random variate.
 */
class DelayMatrixGaussian(future:Real[_,_]?, futureUpdate:Boolean,
    M:Real[_,_], U:Real[_,_], V:Real[_,_]) <
    DelayValue<Real[_,_]>(future, futureUpdate) {
  /**
   * Mean.
   */
  M:Real[_,_] <- M;
  
  /**
   * Within-row covariance.
   */
  U:Real[_,_] <- U;

  /**
   * Within-column covariance.
   */
  V:Real[_,_] <- V;

  function rows() -> Integer {
    return global.rows(M);
  }
  
  function columns() -> Integer {
    return global.columns(M);
  }

  function simulate() -> Real[_,_] {
    return simulate_matrix_gaussian(M, U, V);
  }
  
  function logpdf(X:Real[_,_]) -> Real {
    return logpdf_matrix_gaussian(X, M, U, V);
  }

  function update(X:Real[_,_]) {
    //
  }

  function downdate(X:Real[_,_]) {
    //
  }

  function pdf(X:Real[_,_]) -> Real {
    return pdf_matrix_gaussian(X, M, U, V);
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "MatrixGaussian");
    buffer.set("M", M);
    buffer.set("U", U);
    buffer.set("V", V);
  }
}

function DelayMatrixGaussian(future:Real[_,_]?, futureUpdate:Boolean,
    M:Real[_,_], U:Real[_,_], V:Real[_,_]) -> DelayMatrixGaussian {
  m:DelayMatrixGaussian(future, futureUpdate, M, U, V);
  return m;
}
