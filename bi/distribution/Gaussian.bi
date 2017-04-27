import math;
import random;
import io;

/**
 * Gaussian distribution.
 */
class Gaussian {
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

  function simulate() -> Real {
    cpp {{
    return std::normal_distribution<double>(μ, σ)(rng);
    }}
  }

  function observe(x:Real) -> Real {
    return -0.5*pow((x - this.μ)/this.σ, 2.0) - log(sqrt(2.0*π)*this.σ);
  }
  
  function smooth(μ:Real, σ:Real) {
    this.μ <- μ;
    this.σ <- σ;
  }
}

/**
 * Gaussian distribution with conjugate prior over the mean.
 */
class GaussianWithConjugateMean < Gaussian {
  m0:Gaussian;
  a:Real;

  function create(m0:Gaussian, a:Real) {
    this.μ <- m0.μ;
    this.σ <- sqrt(pow(m0.σ, 2.0) + pow(a, 2.0));
    this.m0 <- m0;
    this.a <- a;
  }
  
  function observe(x:Real) -> Real {
    σ2_0:Real <- pow(this.m0.σ, 2.0);
    λ_0:Real <- 1.0/σ2_0;
    σ2:Real <- pow(this.a, 2.0);
    λ:Real <- 1.0/σ2;
    σ2_1:Real <- 1.0/(λ_0 + λ);
    μ_1:Real <- (this.m0.μ*λ_0 + x*λ)*σ2_1;
    σ_1:Real <- sqrt(σ2_1);
    
    this.m0.smooth(μ_1, σ_1);
    return -0.5*pow((x - this.μ)/this.σ, 2.0) - log(sqrt(2.0*π)*this.σ);
  }
  
  function smooth(μ:Real, σ:Real) {
    this.μ <- μ;
    this.σ <- σ;
  }
}

/**
 * Gaussian distribution that is scalar multiple of another.
 */
class GaussianMultiple < Gaussian {
  a:Real;
  m0:Gaussian;

  function create(a:Real, m0:Gaussian) {
    this.μ <- m0.μ*a;
    this.σ <- m0.σ*abs(a);
    this.a <- a;
    this.m0 <- m0;
  }
  
  function observe(x:Real) -> Real {
    return this.m0.observe(x/abs(this.a)) - log(abs(this.a));
  }
  
  function smooth(μ:Real, σ:Real) {
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
function (x:Real ~> m:Gaussian) -> Real {
  return m.observe(x);
}

/**
 * Initialise.
 */
function (x:Gaussian ~ m:Gaussian) {

}

/**
 * Expressions
 * -----------
 */
/**
 * Create.
 */
function Gaussian(μ:Real, σ:Real) -> Gaussian {
  m:Gaussian;
  m.create(μ, σ);
  return m;
}

/**
 * Marginalise.
 */
function Gaussian(μ:Gaussian, σ:Real) -> Gaussian {
  m:GaussianWithConjugateMean;
  m.create(μ, σ);
  return m;
}

/**
 * Multiply by scalar on left.
 */
function (a:Real*m0:Gaussian) -> Gaussian {
  m1:GaussianMultiple;
  m1.create(a, m0);
  return m1;
}

/**
 * Multiply by scalar on right.
 */
function (m0:Gaussian*a:Real) -> Gaussian {
  m1:GaussianMultiple;
  m1.create(a, m0);
  return m1;
}
