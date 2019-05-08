/*
 * Delayed regression random variate.
 */
final class DelayRegression(future:Real[_]?, futureUpdate:Boolean,
    W:Real[_,_], σ2:Real[_], u:Real[_]) < DelayValue<Real[_]>(
    future, futureUpdate) {
  /**
   * Weights.
   */
  W:Real[_,_] <- W;

  /**
   * Variances.
   */
  σ2:Real[_] <- σ2;

  /**
   * Input.
   */
  u:Real[_] <- u;

  function simulate() -> Real[_] {
    return simulate_regression(W, σ2, u);
  }
  
  function observe(x:Real[_]) -> Real {
    return observe_regression(x, W, σ2, u);
  }

  function update(x:Real[_]) {
    //
  }

  function downdate(x:Real[_]) {
    //
  }

  function pdf(x:Real[_]) -> Real {
    return exp(observe(x));
  }

  function write(buffer:Buffer) {
    buffer.set("W", W);
    buffer.set("σ2", σ2);
  }
}

function DelayRegression(future:Real[_]?, futureUpdate:Boolean,
    W:Real[_,_], σ2:Real[_], u:Real[_]) -> DelayRegression {
  m:DelayRegression(future, futureUpdate, W, σ2, u);
  return m;
}
