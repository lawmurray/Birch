/*
 * Test beta-Bernoulli conjugacy.
 */
program test_beta_bernoulli(N:Integer <- 10000) {
  m:TestBetaBernoulli;
  test_conjugacy(m, N, 2);
}

class TestBetaBernoulli < Model {
  ρ:Random<Real>;
  x:Random<Boolean>;
  
  function initialize() {
    //
  }
  
  fiber simulate() -> Event {
    α:Real <- simulate_uniform(1.0, 10.0);
    β:Real <- simulate_uniform(1.0, 10.0);
    ρ ~ Beta(α, β);
    x ~ Bernoulli(ρ);
  }
  
  function forward() -> Real[_] {
    y:Real[2];
    y[1] <- ρ.value();
    assert !x.hasValue();
    if (x.value()) {
      y[2] <- 1.0;
    } else {
      y[2] <- 0.0;
    }
    
    return y;
  }

  function backward() -> Real[_] {
    y:Real[2];
    if (x.value()) {
      y[2] <- 1.0;
    } else {
      y[2] <- 0.0;
    }
    assert !ρ.hasValue();
    y[1] <- ρ.value();
    
    return y;
  }
  
  function marginal() -> Distribution<Boolean> {
    return x.distribution()!.graft();
  }
}
