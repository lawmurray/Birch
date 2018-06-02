/*
 * Delayed delta function on a sum of two bounded discrete random variates.
 */
class DelayAddBoundedDiscreteDelta(x:Random<Integer>&, x1:DelayBoundedDiscrete,
    x2:DelayBoundedDiscrete) < DelayBoundedDiscrete(x, x1.l + x2.l,
    x1.u + x2.u) {
  /**
   * First discrete random variate.
   */
  x1:DelayBoundedDiscrete <- x1;

  /**
   * Second discrete random variate.
   */
  x2:DelayBoundedDiscrete <- x2;

  function simulate() -> Integer {
    return simulate_delta(x1.simulate() + x2.simulate());
  }
  
  function observe(x:Integer) -> Real {
    l:Integer <- max(x1.l, x - x2.u);
    u:Integer <- min(x1.u, x - x2.l);
    
    /* distribution over possible pairs that produce the observed sum */
    z:Real[u - l + 1];
    Z:Real <- 0.0;
    n:Integer;
    for (n in l..u) {
      z[n - l + 1] <- x1.pmf(n)*x2.pmf(x - n);
      Z <- Z + z[n - l + 1];
    }
    
    /* choose which pair and observe */
    n <- simulate_categorical(z, Z) + l - 1;
    return x1.observe(n) + x1.observe(x - n);
  }

  function pmf(x:Integer) -> Real {
    l:Integer <- max(x1.l, x - x2.u);
    u:Integer <- min(x1.u, x - x2.l);
    P:Real <- 0.0;
    for (n:Integer in l..u) {
      P <- P + x1.pmf(n)*x2.pmf(x - n);
    }
    return P;
  }

  function cdf(x:Integer) -> Real {
    P:Real <- 0.0;
    for (n:Integer in l..x) {
      P <- P + pmf(n);
    }
  }

  function lower() -> Integer? {
    return x1.l + x2.l;
  }
  
  function upper() -> Integer? {
    return x1.u + x2.u;
  }
}

function DelayAddBoundedDiscreteDelta(x:Random<Integer>&,
    x1:DelayBoundedDiscrete, x2:DelayBoundedDiscrete) ->
    DelayAddBoundedDiscreteDelta {
  m:DelayAddBoundedDiscreteDelta(x, x1, x2);
  x1.setChild(m);
  x2.setChild(m);
  return m;
}
