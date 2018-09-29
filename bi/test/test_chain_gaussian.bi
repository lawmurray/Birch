/*
 * Test a chain of conjugate Gaussians.
 */
program test_chain_gaussian(N:Integer <- 10000) {
  X1:Real[N,5];
  X2:Real[N,5];
  μ:Real <- simulate_uniform(-10.0, 10.0);
  σ2:Real[_] <- [
      simulate_uniform(0.0, 10.0),
      simulate_uniform(0.0, 10.0),
      simulate_uniform(0.0, 10.0),
      simulate_uniform(0.0, 10.0),
      simulate_uniform(0.0, 10.0)
    ];
 
  /* simulate forward */
  for i:Integer in 1..N {
    m:TestChainGaussian(μ, σ2);
    m.initialize();
    X1[i,1..5] <- m.forward();
  }

  /* simulate backward */
  for i:Integer in 1..N {
    m:TestChainGaussian(μ, σ2);
    m.initialize();
    X2[i,1..5] <- m.backward();
  }
  
  /* test result */
  if (!pass(X1, X2)) {
    exit(1);
  }
}

class TestChainGaussian(μ:Real, σ2:Real[_]) {
  μ:Real <- μ;
  σ2:Real[_] <- σ2;
  x:Random<Real>[5];
  
  function initialize() {
    x[1] ~ Gaussian(μ, σ2[1]);
    x[2] ~ Gaussian(x[1], σ2[2]);
    x[3] ~ Gaussian(x[2], σ2[3]);
    x[4] ~ Gaussian(x[3], σ2[4]);
    x[5] ~ Gaussian(x[4], σ2[5]);
  }
  
  function forward() -> Real[_] {
    y:Real[5];
    for i:Integer in 1..5 {
      assert !x[i].hasValue();
      y[i] <- x[i].value();
    }
    return y;
  }

  function backward() -> Real[_] {
    y:Real[5];
    for i:Integer in 0..4 {
      assert !x[5 - i].hasValue();
      y[5 - i] <- x[5 - i].value();
    }
    return y;
  }
}
