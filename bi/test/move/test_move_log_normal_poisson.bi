/*
 * Test move on a log-normal-Poisson pair.
 */
program test_move_log_normal_poisson(N:Integer <- 10000) {
  m:TestLogNormalPoisson;
  delay.handle(m.simulate());

  /* simulate forward */
  X1:Real[N,2];
  for n in 1..N {
    auto m' <- clone<TestLogNormalPoisson>(m);
    X1[n,1..2] <- m'.forward();
  }

  /* simulate backward */
  X2:Real[N,2];
  for n in 1..N {
    auto m' <- clone<TestLogNormalPoisson>(m);
    X2[n,1..2] <- m'.backward();
  }
  
  /* test result */
  if !pass(X1, X2) {
    exit(1);
  }
}

class TestLogNormalPoisson < Model {
  λ:Random<Real>;
  x:Random<Integer>;
  
  fiber simulate() -> Event {
    auto μ <- simulate_uniform(0.0, 1.0);
    auto σ2 <- simulate_uniform(1.0, 2.0);
    
    λ ~ Gaussian(μ, σ2);
    x ~ Poisson(exp(λ));
  }
  
  function forward() -> Real[_] {
    y:Real[2];
    y[1] <- λ.value();
    y[2] <- x.value();
    return y;
  }

  function backward() -> Real[_] {
    y:Real[2];
    y[2] <- x.value();
    y[1] <- λ.value();
    return y;
  }
  
  function marginal() -> Distribution<Integer> {
    return x.distribution();
  }
}
