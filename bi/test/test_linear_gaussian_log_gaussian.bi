/*
 * Test linear Gaussian-Gaussian conjugacy.
 */
program test_linear_gaussian_log_gaussian(N:Integer <- 10000) {
  X1:Real[N,2];
  X2:Real[N,2];
  a:Real <- simulate_uniform(-2.0, 2.0);
  c:Real <- simulate_uniform(-10.0, 10.0);
  μ_0:Real <- simulate_uniform(-10.0, 10.0);
  σ2_0:Real <- simulate_uniform(0.0, 2.0);
  σ2_1:Real <- simulate_uniform(0.0, 2.0);
 
  /* simulate forward */
  for i:Integer in 1..N {
    m:TestLinearGaussianLogGaussian(a, μ_0, σ2_0, c, σ2_1);
    m.initialize();
    X1[i,1..2] <- m.forward();
  }

  /* simulate backward */
  for i:Integer in 1..N {
    m:TestLinearGaussianLogGaussian(a, μ_0, σ2_0, c, σ2_1);
    m.initialize();
    X2[i,1..2] <- m.backward();
  }
  
  /* test result */
  if (!pass(X1, X2)) {
    exit(1);
  }
}

class TestLinearGaussianLogGaussian(a:Real, μ_0:Real, σ2_0:Real, c:Real, σ2_1:Real) {
  a:Real <- a;
  μ_0:Real <- μ_0;
  σ2_0:Real <- σ2_0;
  c:Real <- c;
  σ2_1:Real <- σ2_1;
  
  μ_1:Random<Real>;
  x:Random<Real>;
  
  function initialize() {
    μ_1 ~ Gaussian(μ_0, σ2_0);
    x ~ LogGaussian(a*μ_1 + c, σ2_1);
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
}
