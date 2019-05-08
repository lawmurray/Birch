/*
 * Delayed ridge-regression random variate.
 */
final class DelayRidgeRegression(future:Real[_]?, futureUpdate:Boolean,
    θ:DelayRidge, u:Real[_]) < DelayValue<Real[_]>(future, futureUpdate) {
  /**
   * Parameters.
   */
  θ:DelayRidge& <- θ;

  /**
   * Input.
   */
  u:Real[_] <- u;

  function simulate() -> Real[_] {
    return simulate_ridge_regression(θ!.N, θ!.Λ, θ!.α, θ!.γ, u);
  }
  
  function observe(x:Real[_]) -> Real {
    return observe_ridge_regression(x, θ!.N, θ!.Λ, θ!.α, θ!.γ, u);
  }

  function update(x:Real[_]) {
    (θ!.N, θ!.Λ, θ!.α, θ!.γ) <- update_ridge_regression(x, θ!.N, θ!.Λ, θ!.α, θ!.γ, u);
  }

  function downdate(x:Real[_]) {
    (θ!.N, θ!.Λ, θ!.α, θ!.γ) <- downdate_ridge_regression(x, θ!.N, θ!.Λ, θ!.α, θ!.γ, u);
  }

  function pdf(x:Real[_]) -> Real {
    return exp(observe(x));
  }

  function write(buffer:Buffer) {
    buffer.set(value());
  }
}

function DelayRidgeRegression(future:Real[_]?, futureUpdate:Boolean,
    θ:DelayRidge, u:Real[_]) -> DelayRidgeRegression {
  m:DelayRidgeRegression(future, futureUpdate, θ, u);
  return m;
}
