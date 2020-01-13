/*
 * Test beta-binomial conjugacy.
 */
program test_beta_binomial(N:Integer <- 10000) {  
  m:TestBetaBinomial;
  playDelay.handle(m.simulate());
 
  /* simulate forward */
  X1:Real[N,2];
  for i in 1..N {
    auto m' <- clone<TestBetaBinomial>(m);
    X1[i,1..2] <- m'.forward();
  }

  /* simulate backward */
  X2:Real[N,2];
  for i in 1..N {
    auto m' <- clone<TestBetaBinomial>(m);
    X2[i,1..2] <- m'.backward();
  }
  
  /* test result */
  if !pass(X1, X2) {
    exit(1);
  }
}

class TestBetaBinomial < Model {
  ρ:Random<Real>;
  x:Random<Integer>;
  
  fiber simulate() -> Event {
    auto n <- simulate_uniform_int(1, 100);
    auto α <- simulate_uniform(1.0, 10.0);
    auto β <- simulate_uniform(1.0, 10.0);
    
    ρ ~ Beta(α, β);
    x ~ Binomial(n, ρ);
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
