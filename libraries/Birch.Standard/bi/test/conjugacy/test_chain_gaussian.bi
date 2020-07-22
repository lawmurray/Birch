/*
 * Test a chain of conjugate Gaussians.
 */
program test_chain_gaussian(N:Integer <- 10000) {
  m:TestChainGaussian;
  test_conjugacy(m, N, 5);
}

class TestChainGaussian < Model {
  x:Random<Real>[5];
  μ:Real;
  σ2:Real[_];
  
  function initialize() {
    μ <- simulate_uniform(-10.0, 10.0);
    σ2 <- [
      simulate_uniform(0.0, 10.0),
      simulate_uniform(0.0, 10.0),
      simulate_uniform(0.0, 10.0),
      simulate_uniform(0.0, 10.0),
      simulate_uniform(0.0, 10.0)
    ];
  }
  
  fiber simulate() -> Event {
    x[1] ~ Gaussian(μ, σ2[1]);
    x[2] ~ Gaussian(x[1], σ2[2]);
    x[3] ~ Gaussian(x[2], σ2[3]);
    x[4] ~ Gaussian(x[3], σ2[4]);
    x[5] ~ Gaussian(x[4], σ2[5]);
  }
  
  function forward() -> Real[_] {
    y:Real[5];
    for i in 1..5 {
      assert !x[i].hasValue();
      y[i] <- x[i].value();
    }
    return y;
  }

  function backward() -> Real[_] {
    y:Real[5];
    for i in 0..4 {
      assert !x[5 - i].hasValue();
      y[5 - i] <- x[5 - i].value();
    }
    return y;
  }
}
