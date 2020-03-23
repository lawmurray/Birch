/*
 * Test beta-negative-binomial conjugacy.
 */
program test_beta_negative_binomial(N:Integer <- 10000) {
  m:TestBetaNegativeBinomial;
  playDelay.handle(m.simulate());
 
  /* simulate forward */
  X1:Real[N,2];
  for n in 1..N {
    auto m' <- clone(m);
    X1[n,1..2] <- m'.forward();
  }

  /* simulate backward */
  X2:Real[N,2];
  for n in 1..N {
    auto m' <- clone(m);
    X2[n,1..2] <- m'.backward();
  }
  
  /* test result */
  if !pass(X1, X2) {
    exit(1);
  }
}

class TestBetaNegativeBinomial < Model {
  ρ:Random<Real>;
  x:Random<Integer>;
  
  fiber simulate() -> Event {
    k:Integer <- simulate_uniform_int(1, 100);
    α:Real <- simulate_uniform(1.0, 100.0);
    β:Real <- simulate_uniform(1.0, 100.0);
  
    ρ ~ Beta(α, β);
    x ~ NegativeBinomial(k, ρ);
  }
  
  function forward() -> Real[_] {
    y:Real[2];    
    y[1] <- ρ.value();
    assert !x.hasValue();
    y[2] <- x.value();
    return y;
  }

  function backward() -> Real[_] {
    y:Real[2];    
    y[2] <- x.value();
    assert !ρ.hasValue();
    y[1] <- ρ.value();
    return y;
  }
  
  function marginal() -> Distribution<Integer> {
    return x.distribution()!.graft();
  }
}
