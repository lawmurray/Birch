/*
 * Test linear-discrete-delta conjugacy.
 */
program test_linear_discrete_delta(N:Integer <- 10000) {
  m:TestLinearDiscreteDelta;
  delay.handle(m.simulate());
     
  /* simulate forward */
  X1:Real[N,2];
  for i in 1..N {
    auto m' <- clone<TestLinearDiscreteDelta>(m);
    X1[i,1..2] <- m'.forward();
  }

  /* simulate backward */
  X2:Real[N,2];
  for i in 1..N {
    auto m' <- clone<TestLinearDiscreteDelta>(m);
    X2[i,1..2] <- m'.backward();
  }
  
  /* test result */
  if!pass(X1, X2) {
    exit(1);
  }
}

class TestLinearDiscreteDelta < Model {
  ρ:Random<Real>;
  x:Random<Integer>;
  y:Random<Integer>;
  
  fiber simulate() -> Event {
    a:Integer <- 2*simulate_uniform_int(0, 1) - 1;
    n:Integer <- simulate_uniform_int(1, 100);
    α:Real <- simulate_uniform(0.0, 10.0);
    β:Real <- simulate_uniform(0.0, 10.0);
    c:Integer <- simulate_uniform_int(0, 100);

    ρ ~ Beta(α, β);
    x ~ Binomial(n, ρ);
    y ~ Delta(a*x + c);
  }
  
  function forward() -> Real[_] {
    z:Real[2];    
    assert !ρ.hasValue();
    z[1] <- ρ.value();
    assert !x.hasValue();
    z[2] <- x.value();
    return z;
  }

  function backward() -> Real[_] {
    z:Real[2];
    assert !x.hasValue();
    z[2] <- x.value();
    assert !ρ.hasValue();
    z[1] <- ρ.value();
    return z;
  }
  
  function marginal() -> Distribution<Integer> {
    return y.distribution()!;
  }
}
