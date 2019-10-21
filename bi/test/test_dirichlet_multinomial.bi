/*
 * Test Dirichlet-multinomial conjugacy.
 */
program test_dirichlet_multinomial(N:Integer <- 10000) {
  X1:Real[N,10];
  X2:Real[N,10];
  n:Integer <- simulate_uniform_int(100, 500);
  α:Real[5];
  for auto i in 1..5 {
    α[i] <- simulate_uniform(1.0, 10.0);
  }
 
  /* simulate forward */
  for auto i in 1..N {
    m:TestDirichletMultinomial(n, α);
    m.play();
    X1[i,1..10] <- m.forward();
  }

  /* simulate backward */
  for auto i in 1..N {
    m:TestDirichletMultinomial(n, α);
    m.play();
    X2[i,1..10] <- m.backward();
  }
  
  /* test result */
  if (!pass(X1, X2)) {
    exit(1);
  }
}

class TestDirichletMultinomial(n:Integer, α:Real[_]) < Model {
  n:Integer <- n;
  α:Real[_] <- α; 
  ρ:Random<Real[_]>;
  x:Random<Integer[_]>;
  
  fiber simulate() -> Event {
    ρ ~ Dirichlet(α);
    x ~ Multinomial(n, ρ);
  }
  
  function forward() -> Real[_] {
    D:Integer <- length(α);
    y:Real[2*D];
    y[1..D] <- ρ.value();
    assert !x.hasValue();
    y[D+1..2*D] <- x.value();
    return y;
  }

  function backward() -> Real[_] {
    D:Integer <- length(α);
    y:Real[2*D];
    y[D+1..2*D] <- x.value();
    assert !ρ.hasValue();
    y[1..D] <- ρ.value();
    return y;
  }
}
