/*
 * Delayed delta function on a difference of two bounded discrete random
 * variates.
 */
class DelaySubtractBoundedDiscreteDelta(x:Random<Integer>&,
    x1:DelayBoundedDiscrete, x2:DelayBoundedDiscrete) < DelayBoundedDiscrete(
    x, x1.l - x2.u, x1.u - x2.l) {
  /**
   * First discrete random variate.
   */
  x1:DelayBoundedDiscrete <- x1;

  /**
   * Second discrete random variate.
   */
  x2:DelayBoundedDiscrete <- x2;

  function simulate() -> Integer {
    return simulate_delta(x1.simulate() - x2.simulate());
  }
  
  function observe(x:Integer) -> Real {
    l:Integer <- max(x1.l, x2.l - x);
    u:Integer <- min(x1.u, x2.u - x);
    
    /* distribution over possible pairs that produce the observed sum */
    z:Real[u - l + 1];
    Z:Real <- 0.0;
    n:Integer;
    for (n in l..u) {
      z[n - l + 1] <- x1.pmf(n)*x2.pmf(n - x);
      Z <- Z + z[n - l + 1];
    }
    
    /* choose which pair and observe */
    n <- simulate_categorical(z, Z) + l - 1;
    return x1.observe(n) + x2.observe(x - n);
  }

  function pmf(x:Integer) -> Real {
    l:Integer <- max(x1.l, x2.l - x);
    u:Integer <- min(x1.u, x2.u - x);
    P:Real <- 0.0;
    for (n:Integer in l..u) {
      P <- P + x1.pmf(n)*x2.pmf(n - x);
    }
    return P;
  }

  function cdf(x:Integer) -> Real {
    P:Real <- 0.0;
    for (n:Integer in l..x) {
      P <- P + pmf(n);
    }
  }
}

function DelaySubtractBoundedDiscreteDelta(x:Random<Integer>&,
    x1:DelayBoundedDiscrete, x2:DelayBoundedDiscrete) ->
    DelaySubtractBoundedDiscreteDelta {
  m:DelaySubtractBoundedDiscreteDelta(x, x1, x2);
  x1.setChild(m);
  x2.setChild(m);
  return m;
}
