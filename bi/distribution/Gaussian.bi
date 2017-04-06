import math;
import random;
import io;

/**
 * Gaussian distribution.
 */
model Gaussian {
  /**
   * Mean.
   */
  μ:Real;
  
  /**
   * Standard deviation.
   */
  σ:Real;

  function create(μ:Real, σ:Real) {
    this.μ <- μ;
    this.σ <- σ;
  }

  function simulate() -> x:Real {
    cpp {{
    x = std::normal_distribution<double>(μ, σ)(rng);
    }}
  }

  virtual function observe(x:Real) -> l:Real {
    l <- -0.5*pow((x - this.μ)/this.σ, 2.0) - log(sqrt(2.0*π)*this.σ);
  }
  
  virtual function smooth(μ:Real, σ:Real) {
    this.μ <- μ;
    this.σ <- σ;
  }
}

/**
 * Gaussian distribution with conjugate prior over the mean.
 */
model GaussianWithConjugateMean < Gaussian {
  m0:Gaussian;
  a:Real;

  function create(m0:Gaussian, a:Real) {
    this.μ <- m0.μ;
    this.σ <- sqrt(pow(m0.σ, 2.0) + pow(a, 2.0));
    this.m0 <- m0;
    this.a <- a;
  }
  
  virtual function observe(x:Real) -> l:Real {
    σ2_0:Real <- pow(this.m0.σ, 2.0);
    λ_0:Real <- 1.0/σ2_0;
    σ2:Real <- pow(this.a, 2.0);
    λ:Real <- 1.0/σ2;
    σ2_1:Real <- 1.0/(λ_0 + λ);
    μ_1:Real <- (this.m0.μ*λ_0 + x*λ)*σ2_1;
    σ_1:Real <- sqrt(σ2_1);
    
    this.m0.smooth(μ_1, σ_1);
    l <- -0.5*pow((x - this.μ)/this.σ, 2.0) - log(sqrt(2.0*π)*this.σ);
  }
  
  virtual function smooth(μ:Real, σ:Real) {
    this.μ <- μ;
    this.σ <- σ;
  }
}

/**
 * Gaussian distribution that is scalar multiple of another.
 */
model GaussianMultiple < Gaussian {
  a:Real;
  m0:Gaussian;

  virtual function create(a:Real, m0:Gaussian) {
    this.μ <- m0.μ*a;
    this.σ <- m0.σ*abs(a);
    this.a <- a;
    this.m0 <- m0;
  }
  
  virtual function observe(x:Real) -> l:Real {
    l <- this.m0.observe(x/abs(this.a)) - log(abs(this.a));
  }
  
  virtual function smooth(μ:Real, σ:Real) {
    this.m0.smooth(μ/this.a, σ/abs(this.a));
  }
}

/**
 * Operators
 * ---------
 */
/**
 * Simulate.
 */
function (x:Real <~ m:Gaussian) {
  x <- m.simulate();
}

/**
 * Observe.
 */
function (x:Real ~> m:Gaussian) -> l:Real {
  l <- m.observe(x);
}

/**
 * Delay.
 */
function (x:(Real ~ Gaussian) ~ m:Gaussian) {
  cpp{{
  x = m;
  }}
}

/**
 * Expressions
 * -----------
 */
/**
 * Create.
 */
function Gaussian(μ:Real, σ:Real) -> m:Gaussian {
  m.create(μ, σ);
}

/**
 * Marginalise.
 */
function Gaussian(μ:Gaussian, σ:Real) -> m:GaussianWithConjugateMean {
  m.create(μ, σ);
}

/**
 * Multiply by scalar on left.
 */
function (a:Real*m0:Gaussian) -> m1:GaussianMultiple {
  m1.create(a, m0);
}

/**
 * Multiply by scalar on right.
 */
function (m0:Gaussian*a:Real) -> m1:GaussianMultiple {
  m1.create(a, m0);
}
