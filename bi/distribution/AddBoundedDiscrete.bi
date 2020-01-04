/*
 * ed delta function on a sum of two bounded discrete random variates.
 */
final class AddBoundedDiscrete(future:Integer?, futureUpdate:Boolean,
    x1:BoundedDiscrete, x2:BoundedDiscrete) < BoundedDiscrete(
    future, futureUpdate, x1.l + x2.l, x1.u + x2.u) {
  /**
   * First node.
   */
  x1:BoundedDiscrete& <- x1;

  /**
   * Second node.
   */
  x2:BoundedDiscrete& <- x2;
  
  /**
   * Value for which conditional probabilities have been enumerated.
   */
  x:Integer?;
  
  /**
   * If clamped, then the lower bound of `x1`.
   */
  x0:Integer;
  
  /**
   * If clamped, then the probabilities of all possible values of `x1`,
   * starting from `x0`.
   */
  z:Real[_];
  
  /**
   * If clamped, then the sum of `z`.
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
      auto l <- max(x1.l, x - x2.u);
      auto u <- min(x1.u, x - x2.l);
    
      x0 <- l;
      Z <- 0.0;
      if l <= u {
        /* distribution over possible pairs that produce the given sum */
        z <- vector(0.0, u - l + 1);
        for n in l..u {
          z[n - l + 1] <- exp(x1.logpdf(n) + x2.logpdf(x - n));
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
      return simulate_delta(x1.simulate() + x2.simulate());
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
      /* choose a pair with the given sum and clamp parents */
      enumerate(x);
      auto n <- simulate_categorical(z, Z) + x0 - 1;
      x1.clamp(n);
      x2.clamp(x - n);
      alreadyUpdated <- true;
    }
  }

  function cdf(x:Integer) -> Real? {
    auto P <- 0.0;
    for n in l..x {
      P <- P + pdf(n);
    }
    return P;
  }
}

function AddBoundedDiscrete(future:Integer?, futureUpdate:Boolean,
    x1:BoundedDiscrete, x2:BoundedDiscrete) ->
    AddBoundedDiscrete {
  m:AddBoundedDiscrete(future, futureUpdate, x1, x2);
  x1.setChild(m);
  x2.setChild(m);
  return m;
}
