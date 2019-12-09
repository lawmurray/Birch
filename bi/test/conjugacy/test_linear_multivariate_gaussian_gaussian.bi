/*
 * Test multivariate linear Gaussian-Gaussian conjugacy involving dot product.
 */
program test_linear_multivariate_gaussian_gaussian(
    N:Integer <- 10000) {
  m:TestLinearMultivariateGaussianGaussian;
  delay.handle(m.simulate());
 
  /* simulate forward */
  X1:Real[N,6];
  for n in 1..N {
    auto m' <- clone<TestLinearMultivariateGaussianGaussian>(m);
    X1[n,1..6] <- m'.forward();
  }

  /* simulate backward */
  X2:Real[N,6];
  for n in 1..N {
    auto m' <- clone<TestLinearMultivariateGaussianGaussian>(m);
    X2[n,1..6] <- m'.backward();
  }
  
  /* test result */
  if !pass(X1, X2) {
    exit(1);
  }
}

class TestLinearMultivariateGaussianGaussian < Model {
  μ:Random<Real[_]>;
  x:Random<Real>;
  
  fiber simulate() -> Event {
    a:Real[5];
    μ_0:Real[5];
    Σ_0:Real[5,5];
    c:Real;
    σ2_1:Real;

    c <- simulate_uniform(-10.0, 10.0);
    for i in 1..5 {
      a[i] <- simulate_uniform(-2.0, 2.0);
      μ_0[i] <- simulate_uniform(-10.0, 10.0);
      for j in 1..5 {
        Σ_0[i,j] <- simulate_uniform(-2.0, 2.0);
      }
    }
    Σ_0 <- Σ_0*transpose(Σ_0);
    σ2_1 <- pow(simulate_uniform(-2.0, 2.0), 2.0);

    μ ~ Gaussian(μ_0, Σ_0);
    x ~ Gaussian(dot(a, μ) + c, σ2_1);
  }
  
  function forward() -> Real[_] {
    y:Real[6];
    y[1..5] <- μ.value();
    assert !x.hasValue();
    y[6] <- x.value();
    return y;
  }

  function backward() -> Real[_] {
    y:Real[6];
    y[6] <- x.value();
    assert !μ.hasValue();
    y[1..5] <- μ.value();
    return y;
  }
  
  function marginal() -> Distribution<Real> {
    return x.distribution();
  }
}
