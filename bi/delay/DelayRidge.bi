/*
 * Delayed ridge variate.
 */
final class DelayRidge(future:(Real[_,_],Real[_])?, futureUpdate:Boolean,
    M:Real[_,_], Σ:Real[_,_], α:Real, β:Real[_]) <
    DelayValue<(Real[_,_],Real[_])>(future, futureUpdate) {
  /**
   * Common precision, $\Lambda = \Sigma^{-1}$.
   */
  Λ:LLT <- llt(cholinv(Σ));

  /**
   * Precision times mean, $Ν = \Lambda M$.
   */
  N:Real[_,_] <- Λ*M;

  /**
   * Common weight and likelihood covariance shape.
   */
  α:Real <- α;

  /**
   * Covariance scale accumulators,
   * $\gamma = \beta + \frac{1}{2} N^\top \Lambda^{-1} N$.
   */
  γ:Real[_] <- β + 0.5*diagonal(transpose(N)*M);

  function simulate() -> (Real[_,_],Real[_]) {
    return simulate_ridge(N, Λ, α, γ);
  }
  
  function logpdf(x:(Real[_,_],Real[_])) -> Real {   
    W:Real[_,_];
    σ2:Real[_];
    (W, σ2) <- x;
    return logpdf_ridge(W, σ2, N, Λ, α, γ);
  }

  function update(x:(Real[_,_],Real[_])) {
    //
  }

  function downdate(x:(Real[_,_],Real[_])) {
    //
  }

  function pdf(x:(Real[_,_],Real[_])) -> Real {
    return exp(logpdf(x));
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Ridge");
    buffer.set("M", solve(Λ, N));
    buffer.set("Σ", inv(Λ));
    buffer.set("α", α);
    buffer.set("β", γ - 0.5*diagonal(transpose(solve(Λ, N))*N));
  }
}

function DelayRidge(future:(Real[_,_],Real[_])?, futureUpdate:Boolean,
    M:Real[_,_], Σ:Real[_,_], α:Real, β:Real[_]) -> DelayRidge {
  m:DelayRidge(future, futureUpdate, M, Σ, α, β);
  return m;
}
