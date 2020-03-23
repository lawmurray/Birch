/*
 * Test Dirichlet-categorical conjugacy.
 */
program test_dirichlet_categorical(N:Integer <- 10000) { 
  m:TestDirichletCategorical;
  playDelay.handle(m.simulate());

  /* simulate forward */
  X1:Real[N,6];
  for n in 1..N {
    auto m' <- clone(m);
    X1[n,1..6] <- m'.forward();
  }

  /* simulate backward */
  X2:Real[N,6];
  for n in 1..N {
    auto m' <- clone(m);
    X2[n,1..6] <- m'.backward();
  }
  
  /* test result */
  if !pass(X1, X2) {
    exit(1);
  }
}

class TestDirichletCategorical < Model {
  ρ:Random<Real[_]>;
  x:Random<Integer>;
  
  fiber simulate() -> Event {
    α:Real[5];
    for n in 1..5 {
      α[n] <- simulate_uniform(1.0, 10.0);
    }
    ρ ~ Dirichlet(α);
    x ~ Categorical(ρ);
  }
  
  function forward() -> Real[_] {
    y:Real[6];
    y[1..5] <- ρ.value();
     assert !x.hasValue();
    y[6] <- x.value();
    return y;
  }

  function backward() -> Real[_] {
    y:Real[6];
    y[6] <- x.value();
     assert !ρ.hasValue();
    y[1..5] <- ρ.value();
    return y;
  }
  
  function marginal() -> Distribution<Integer> {
    return x.distribution()!.graft();
  }
}
