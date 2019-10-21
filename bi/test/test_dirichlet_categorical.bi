/*
 * Test Dirichlet-categorical conjugacy.
 */
program test_dirichlet_categorical(N:Integer <- 10000) {
  X1:Real[N,6];
  X2:Real[N,6];
  α:Real[5];
  for n:Integer in 1..5 {
    α[n] <- simulate_uniform(1.0, 10.0);
  }
 
  /* simulate forward */
  for auto n in 1..N {
    m:TestDirichletCategorical(α);
    m.play();
    X1[n,1..6] <- m.forward();
  }

  /* simulate backward */
  for auto n in 1..N {
    m:TestDirichletCategorical(α);
    m.play();
    X2[n,1..6] <- m.backward();
  }
  
  /* test result */
  if (!pass(X1, X2)) {
    exit(1);
  }
}

class TestDirichletCategorical(α:Real[_]) < Model {
  α:Real[_] <- α; 
  ρ:Random<Real[_]>;
  x:Random<Integer>;
  
  fiber simulate() -> Event {
    ρ ~ Dirichlet(α);
    x ~ Categorical(ρ);
  }
  
  function forward() -> Real[_] {
    D:Integer <- length(α);
    y:Real[D + 1];
    y[1..D] <- ρ.value();
     assert !x.hasValue();
    y[D + 1] <- x.value();
    return y;
  }

  function backward() -> Real[_] {
    D:Integer <- length(α);
    y:Real[D + 1];
    y[D + 1] <- x.value();
     assert !ρ.hasValue();
    y[1..D] <- ρ.value();
    return y;
  }
}
