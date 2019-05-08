/**
 * Regression model.
 */
final class Regression(θ:Expression<(Real[_,_],Real[_])>,
    x:Expression<Real[_]>) < Distribution<Real[_]> {
  /**
   * Parameters.
   */
  auto θ <- θ;

  /**
   * Input.
   */
  auto x <- x;

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      m:DelayRidge?;
      if (m <- θ.graftRidge())? {
        delay <- DelayRidgeRegression(future, futureUpdate, m!, x);
      } else {
        W:Real[_,_];
        σ2:Real[_];
        (W, σ2) <- θ.value();
        delay <- DelayRegression(future, futureUpdate, W, σ2, x);
      }
    }
  }

  function write(buffer:Buffer) {
    if delay? {
      delay!.write(buffer);
    } else {
      W:Real[_,_];
      σ2:Real[_];
      (W, σ2) <- θ.value();
    
      buffer.set("class", "Regression");
      buffer.set("W", W);
      buffer.set("σ2", W);
      buffer.set("x", x.value());
    }
  }
}

/**
 * Create regression model.
 */
function Regression(θ:Expression<(Real[_,_],Real[_])>,
    x:Expression<Real[_]>) -> Regression {
  m:Regression(θ, x);
  return m;
}

/**
 * Create regression model.
 */
function Regression(θ:Expression<(Real[_,_],Real[_])>, x:Real[_]) ->
    Regression {
  return Regression(θ, Boxed(x));
}

/**
 * Create regression model.
 */
function Regression(θ:(Real[_,_],Real[_]), x:Expression<Real[_]>) ->
    Regression {
  return Regression(Boxed(θ), x);
}

/**
 * Create regression model.
 */
function Regression(θ:(Real[_,_],Real[_]), x:Real[_]) -> Regression {
  return Regression(Boxed(θ), Boxed(x));
}
