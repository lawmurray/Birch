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
  γ:Real[_] <- β + 0.5*diagonal(trans(N)*M);

  function simulate() -> (Real[_,_],Real[_]) {
    //return simulate_ridge(N, Λ, α, γ);
  
    /*auto R <- rows(N);
    auto C <- columns(N);
    auto M <- solve(Λ, N);
    auto Σ <- inv(Λ);
    
    W:Real[R,C];
    σ2:Real[C];
    for auto j in 1..C {
      σ2[j] <- simulate_inverse_gamma(α, β[j]);
      W[1..R,j] <- simulate_multivariate_gaussian(M[1..R,j], Σ*σ2[j]);
    }    
    return (W, σ2);*/
  }
  
  function observe(x:(Real[_,_],Real[_])) -> Real {   
    //return observe_ridge(x, N, Λ, α, γ);
   
    /*auto R <- rows(N);
    auto C <- columns(N);
    auto M <- solve(Λ, N);
    auto Σ <- inv(Λ);
    
    W:Real[_,_];
    σ2:Real[_];
    (W, σ2) <- x;
    auto w <- 0.0;
    for auto j in 1..C {
      w <- w + observe_inverse_gamma(σ2[j], α, β[j]);
      w <- w + observe_multivariate_gaussian(W[1..R,j], M[1..R,j], Σ*σ2[j]);
    }
    return w;*/
  }

  function update(x:(Real[_,_],Real[_])) {
    //
  }

  function downdate(x:(Real[_,_],Real[_])) {
    //
  }

  function pdf(x:(Real[_,_],Real[_])) -> Real {
    return exp(observe(x));
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Ridge");
    buffer.set("M", solve(Λ, N));
    buffer.set("Σ", inv(Λ));
    buffer.set("α", α);
    buffer.set("β", γ - 0.5*diagonal(trans(solve(Λ, N))*N));
  }
}

function DelayRidge(future:(Real[_,_],Real[_])?, futureUpdate:Boolean,
    M:Real[_,_], Σ:Real[_,_], α:Real, β:Real[_]) -> DelayRidge {
  m:DelayRidge(future, futureUpdate, M, Σ, α, β);
  return m;
}
