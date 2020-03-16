/*
 * Grafted delta function on a difference of two bounded discrete random
 * variates.
 */
final class SubtractBoundedDiscrete(x1:BoundedDiscrete, x2:BoundedDiscrete) <
    BoundedDiscrete {
  /**
   * First discrete random variate.
   */
  x1:BoundedDiscrete& <- x1;

  /**
   * Second discrete random variate.
   */
  x2:BoundedDiscrete& <- x2;
  
  /**
   * Value for which conditional probabilities have been enumerated.
   */
  x:Integer?;
  
  /**
   * The lower bound of `x1`.
   */
  x0:Integer;
  
  /**
   * The probabilities of all possible values of `x1`, starting from `x0`.
   */
  z:Real[_];
  
  /**
   * The sum of `z`.
   */
  Z:Real;

  /**
   * Has this node already updated its parents? Nodes of this type have two
   * parent nodes, and because of this, their update() member function is
   * called twice. This flag is a hack to ensure that the actual update is
   * performed only once.
   */
  alreadyUpdated:Boolean <- false;

  function enumerate(x:Integer) {
    if !this.x? || this.x! != x {
      auto l <- max(x1.lower()!, x2.lower()! + x);
      auto u <- min(x1.upper()!, x2.upper()! + x);

      x0 <- l;
      Z <- 0.0;
      if l <= u {
        /* distribution over possible pairs that produce the given diff */
        z <- vector(0.0, u - l + 1);
        for n in l..u {
          z[n - l + 1] <- x1.pdf(n)*x2.pdf(n - x);
          Z <- Z + z[n - l + 1];
        }
      }
      this.x <- x;
    }
  }

  function simulate() -> Integer {
    if value? {
      return simulate_delta(value!);
    } else {
      return simulate_delta(x1.simulate() - x2.simulate());
    }
  }
  
  function logpdf(x:Integer) -> Real {
    if value? {
      return logpdf_delta(x, value!);
    } else {
      enumerate(x);
      return log(Z);
    }
  }

  function update(x:Integer) {
    if !alreadyUpdated {
      /* choose a pair with the given difference and clamp parents */
      enumerate(x);
      auto n <- simulate_categorical(z, Z) + x0 - 1;
      x1.clamp(n);
      x2.clamp(n - x);
      alreadyUpdated <- true;
    }
  }

  function cdf(x:Integer) -> Real? {
    auto P <- 0.0;
    for n in lower()!..x {
      P <- P + pdf(n);
    }
    return P;
  }
  
  function lower() -> Integer? {
    return x1.lower()! - x2.upper()!;
  }
  
  function upper() -> Integer? {
    return x1.upper()! - x2.lower()!;
  }

  function graftFinalize() -> Boolean {
    if !x1.hasValue() && !x2.hasValue() {
      x1.setChild(this);
      x2.setChild(this);
      return true;
    } else {
      return false;
    }
  }
}

function SubtractBoundedDiscrete(x1:BoundedDiscrete, x2:BoundedDiscrete) ->
    SubtractBoundedDiscrete {
  m:SubtractBoundedDiscrete(x1, x2);
  return m;
}
