/*
 * Delayed delta function on a sum of two bounded discrete random variates.
 */
class DelayAddBoundedDiscrete(x:Random<Integer>&, x1:DelayBoundedDiscrete,
    x2:DelayBoundedDiscrete) < DelayBoundedDiscrete(x, x1.l + x2.l,
    x1.u + x2.u) {
  /**
   * First discrete random variate.
   */
  x1:DelayBoundedDiscrete& <- x1;

  /**
   * Second discrete random variate.
   */
  x2:DelayBoundedDiscrete& <- x2;
  
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

  function enumerate(x:Integer) {
    if (!this.x? || this.x! != x) {
      l:Integer <- max(x1!.l, x - x2!.u);
      u:Integer <- min(x1!.u, x - x2!.l);
    
      x0 <- l;
      Z <- 0.0;
      if (l <= u) {
        /* distribution over possible pairs that produce the given sum */
        z <- vector(0.0, u - l + 1);
        for (n:Integer in l..u) {
          z[n - l + 1] <- x1!.pmf(n)*x2!.pmf(x - n);
          Z <- Z + z[n - l + 1];
        }
      }
      this.x <- x;
    }
  }

  function simulate() -> Integer {
    if value? {
      return value!;
    } else {
      return simulate_delta(x1!.simulate() + x2!.simulate());
    }
  }
  
  function observe(x:Integer) -> Real {
    assert !value?;
    enumerate(x);
    return log(Z);
  }
  
  function update(x:Integer) {
    /* choose a pair with the given sum and clamp parents */
    enumerate(x);
    n:Integer <- simulate_categorical(z, Z) + x0 - 1;
    x1!.clamp(n);
    x2!.clamp(x - n);
  }

  function pmf(x:Integer) -> Real {
    l:Integer <- max(x1!.l, x - x2!.u);
    u:Integer <- min(x1!.u, x - x2!.l);
    P:Real <- 0.0;
    for (n:Integer in l..u) {
      P <- P + x1!.pmf(n)*x2!.pmf(x - n);
    }
    return P;
  }

  function cdf(x:Integer) -> Real {
    P:Real <- 0.0;
    for (n:Integer in l..x) {
      P <- P + pmf(n);
    }
    return P;
  }

  function detach() {
    // override as have two parents
    parent <- nil;
    x1!.child <- nil;
    x2!.child <- nil;
  }  
}

function DelayAddBoundedDiscrete(x:Random<Integer>&,
    x1:DelayBoundedDiscrete, x2:DelayBoundedDiscrete) ->
    DelayAddBoundedDiscrete {
  m:DelayAddBoundedDiscrete(x, x1, x2);
  x1.setChild(m);
  x2.setChild(m);
  return m;
}
