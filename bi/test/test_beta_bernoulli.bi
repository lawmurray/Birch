/**
 * Test beta-Bernoulli conjugacy.
 */
program test_beta_bernoulli(N:Integer <- 10000) {
  X1:Real[N,2];
  X2:Real[N,2];
  α:Real <- simulate_uniform(0.0, 100.0);
  β:Real <- simulate_uniform(0.0, 100.0);
 
  /* simulate forward */
  for n:Integer in 1..N {
    m:TestBetaBernoulli(α, β);
    m.initialize();
    X1[n,1..2] <- m.forward();
  }

  /* simulate backward */
  for n:Integer in 1..N {
    m:TestBetaBernoulli(α, β);
    m.initialize();
    X2[n,1..2] <- m.backward();
  }
  
  /* test result */
  if (!pass(X1, X2)) {
    exit(1);
  }
}

class TestBetaBernoulli(α:Real, β:Real) {
  α:Real <- α;
  β:Real <- β;
  ρ:Random<Real>;
  x:Random<Boolean>;
  
  function initialize() {
    ρ ~ Beta(α, β);
    x ~ Bernoulli(ρ);
  }
  
  function forward() -> Real[_] {
    y:Real[2];
    
    y[1] <- ρ.value();
    assert x.isMissing();
    if (x.value()) {
      y[2] <- 1.0;
    } else {
      y[2] <- 0.0;
    }
    
    return y;
  }

  function backward() -> Real[_] {
    y:Real[2];
    
    if (x.value()) {
      y[2] <- 1.0;
    } else {
      y[2] <- 0.0;
    }
    assert ρ.isMissing();
    y[1] <- ρ.value();
    
    return y;
  }
}
