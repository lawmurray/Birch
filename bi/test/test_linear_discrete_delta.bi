/**
 * Test linear-discrete-delta conjugacy.
 */
program test_linear_discrete_delta(N:Integer <- 10000) {
  X1:Real[N,3];
  X2:Real[N,3];
  
  a:Integer <- 2*simulate_uniform_int(0, 1) - 1;
  n:Integer <- simulate_uniform_int(1, 100);
  α:Real <- simulate_uniform(0.0, 10.0);
  β:Real <- simulate_uniform(0.0, 10.0);
  c:Integer <- simulate_uniform_int(0, 100);
 
  /* simulate forward */
  for i:Integer in 1..N {
    m:TestLinearDiscreteDelta(a, n, α, β, c);
    m.initialize();
    X1[i,1..3] <- m.forward();
  }

  /* simulate backward */
  for i:Integer in 1..N {
    m:TestLinearDiscreteDelta(a, n, α, β, c);
    m.initialize();
    X2[i,1..3] <- m.backward();
  }
  
  /* test result */
  if (!pass(X1, X2)) {
    exit(1);
  }
}

class TestLinearDiscreteDelta(a:Integer, n:Integer, α:Real, β:Real,
    c:Integer) {
  a:Integer <- a;
  n:Integer <- n;
  α:Real <- α;
  β:Real <- β;
  c:Integer <- c;
  
  ρ:Random<Real>;
  x:Random<Integer>;
  y:Random<Integer>;
  
  function initialize() {
    ρ ~ Beta(α, β);
    x ~ Binomial(n, ρ);
    y ~ Delta(a*x + c);
  }
  
  function forward() -> Real[_] {
    z:Real[3];    
    z[1] <- ρ.value();
    z[2] <- x.value();
    z[3] <- y.value();
    return z;
  }

  function backward() -> Real[_] {
    z:Real[3];    
    z[3] <- y.value();
    z[2] <- x.value();
    z[1] <- ρ.value();
    return z;
  }
}
