/*
 * Test linear Gaussian-Gaussian conjugacy.
 */
program test_linear_gaussian_gaussian(N:Integer <- 10000) { 
  m:TestLinearGaussianGaussian;
  playDelay.handle(m.simulate());
    
  /* simulate forward */
  X1:Real[N,2];
  for i in 1..N {
    auto m' <- clone<TestLinearGaussianGaussian>(m);
    X1[i,1..2] <- m'.forward();
  }

  /* simulate backward */
  X2:Real[N,2];
  for i in 1..N {
    auto m' <- clone<TestLinearGaussianGaussian>(m);
    X2[i,1..2] <- m'.backward();
  }
  
  /* test result */
  if !pass(X1, X2) {
    exit(1);
  }
}

class TestLinearGaussianGaussian < Model {
  μ_1:Random<Real>;
  x:Random<Real>;
  
  fiber simulate() -> Event {
    a:Real <- simulate_uniform(-2.0, 2.0);
    c:Real <- simulate_uniform(-10.0, 10.0);
    μ_0:Real <- simulate_uniform(-10.0, 10.0);
    σ2_0:Real <- simulate_uniform(0.0, 2.0);
    σ2_1:Real <- simulate_uniform(0.0, 2.0);

    μ_1 ~ Gaussian(μ_0, σ2_0);
    x ~ Gaussian(a*μ_1 + c, σ2_1);
  }
  
  function forward() -> Real[_] {
    y:Real[2];
    y[1] <- μ_1.value();
    assert !x.hasValue();
    y[2] <- x.value();
    return y;
  }

  function backward() -> Real[_] {
    y:Real[2];
    y[2] <- x.value();
    assert !μ_1.hasValue();
    y[1] <- μ_1.value();
    return y;
  }
  
  function marginal() -> Distribution<Real> {
    return x.distribution()!.graft();
  }
}
