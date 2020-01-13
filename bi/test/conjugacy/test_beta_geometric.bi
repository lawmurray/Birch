/*
 * Test beta-geometric conjugacy.
 */
program test_beta_geometric(N:Integer <- 10000) {
  X1:Real[N,2];
  X2:Real[N,2];
  α:Real <- simulate_uniform(1.0, 100.0);
  β:Real <- simulate_uniform(1.0, 100.0);
 
  /* simulate forward */
  for i in 1..N {
    m:TestBetaGeometric(α, β);
    playDelay.handle(m.simulate());
    X1[i,1..2] <- m.forward();
  }

  /* simulate backward */
  for i in 1..N {
    m:TestBetaGeometric(α, β);
    playDelay.handle(m.simulate());
    X2[i,1..2] <- m.backward();
  }
  
  /* test result */
  if !pass(X1, X2) {
    exit(1);
  }
}

class TestBetaGeometric(α:Real, β:Real) < Model {
  α:Real <- α;
  β:Real <- β;
  ρ:Random<Real>;
  x:Random<Integer>;
  
  fiber simulate() -> Event {
    ρ ~ Beta(α, β);
    x ~ Geometric(ρ);
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
}
