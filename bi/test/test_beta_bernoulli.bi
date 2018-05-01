/**
 * Test of beta-Bernoulli conjugacy.
 */
program test_beta_bernoulli(N:Integer <- 1000) {
  X1:Real[N,2];
  X2:Real[N,2];
 
  /* simulate forward */
  for n:Integer in 1..N {
    m:TestBetaBernoulli;
    m.simulate();
    X1[n,1] <- m.ρ.value();
    if (m.x.value()) {
      X1[n,2] <- 1.0;
    } else {
      X1[n,2] <- 0.0;
    }
  }

  /* simulate backward */
  for n:Integer in 1..N {
    m:TestBetaBernoulli;
    m.simulate();
    if (m.x.value()) {
      X2[n,2] <- 1.0;
    } else {
      X2[n,2] <- 0.0;
    }
    X2[n,1] <- m.ρ.value();
  }
  
  /* compare */
  δ:Real;
  ε:Real;
  (δ, ε) <- compare(X1, X2);  
  stderr.print(δ + " vs " + ε + "\n");
}

class TestBetaBernoulli {
  ρ:Random<Real>;
  x:Random<Boolean>;
  
  function simulate() {
    ρ ~ Beta(1.0, 1.0);
    x ~ Bernoulli(ρ);
  }
}
