/*
 * Test a chain of conjugate multivariate Gaussians.
 */
program test_chain_multivariate_gaussian(N:Integer <- 10000) {
  X1:Real[N,15];
  X2:Real[N,15];
  
  μ:Real[3];
  Σ:Real[3,3];

  for i in 1..3 {
    μ[i] <- simulate_uniform(-10.0, 10.0);
    for j in 1..3 {
      Σ[i,j] <- simulate_uniform(-2.0, 2.0);
    }
  }
  Σ <- Σ*transpose(Σ);
 
  /* simulate forward */
  for i in 1..N {
    m:TestChainMultivariateGaussian(μ, Σ);
    playDelay.handle(m.simulate());
    X1[i,1..15] <- m.forward();
  }

  /* simulate backward */
  for i in 1..N {
    m:TestChainMultivariateGaussian(μ, Σ);
    playDelay.handle(m.simulate());
    X2[i,1..15] <- m.backward();
  }
  
  /* test result */
  if !pass(X1, X2) {
    exit(1);
  }
}

class TestChainMultivariateGaussian(μ:Real[_], Σ:Real[_,_]) < Model {
  μ:Real[_] <- μ;
  Σ:Real[_,_] <- Σ;
  x:Random<Real[_]>[5];
  
  fiber simulate() -> Event {
    x[1] ~ Gaussian(μ, Σ);
    x[2] ~ Gaussian(x[1], Σ);
    x[3] ~ Gaussian(x[2], Σ);
    x[4] ~ Gaussian(x[3], Σ);
    x[5] ~ Gaussian(x[4], Σ);
  }
  
  function forward() -> Real[_] {
    y:Real[15];
    for i in 1..5 {
      assert !x[i].hasValue();
      y[(i-1)*3+1..i*3] <- x[i].value();
    }
    return y;
  }

  function backward() -> Real[_] {
    y:Real[15];
    for i in 0..4 {
      assert !x[5-i].hasValue();
      y[(4-i)*3+1..(5-i)*3] <- x[5-i].value();
    }
    return y;
  }
}
