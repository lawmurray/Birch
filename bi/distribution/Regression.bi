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

  function valueForward() -> Real[_] {
    assert !delay?;
    W:Real[_,_];
    σ2:Real[_];
    (W, σ2) <- θ.value();
    return simulate_regression(W, σ2, x);
  }

  function observeForward(y:Real[_]) -> Real {
    assert !delay?;
    W:Real[_,_];
    σ2:Real[_];
    (W, σ2) <- θ.value();
    return logpdf_regression(y, W, σ2, x);
  }

  function graft(force:Boolean) {
    if delay? {
      delay!.prune();
    } else {
      m:DelayRidge?;
      if (m <- θ.graftRidge())? {
        delay <- DelayRidgeRegression(future, futureUpdate, m!, x);
      } else if force {
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
