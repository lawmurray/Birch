/*
 * Test beta-Bernoulli conjugacy.
 */
program test_beta_bernoulli(N:Integer <- 10000) {
  X1:Real[N,2];
  X2:Real[N,2];
  α:Real <- simulate_uniform(0.0, 10.0);
  β:Real <- simulate_uniform(0.0, 10.0);
 
  /* simulate forward */
  for auto n in 1..N {
    m:TestBetaBernoulli(α, β);
    m.play();
    X1[n,1..2] <- m.forward();
  }

  /* simulate backward */
  for auto n in 1..N {
    m:TestBetaBernoulli(α, β);
    m.play();
    X2[n,1..2] <- m.backward();
  }
  
  /* test result */
  if (!pass(X1, X2)) {
    exit(1);
  }
}

class TestBetaBernoulli(α:Real, β:Real) < Model {
  α:Real <- α;
  β:Real <- β;
  ρ:Random<Real>;
  x:Random<Boolean>;
  
  fiber simulate() -> Event {
    ρ ~ Beta(α, β);
    x ~ Bernoulli(ρ);
  }
  
  function forward() -> Real[_] {
    y:Real[2];
    
    y[1] <- ρ.value();
    assert !x.hasValue();
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
    assert !ρ.hasValue();
    y[1] <- ρ.value();
    
    return y;
  }
}
