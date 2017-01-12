import math;
import random;

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
}

/**
 * Create.
 */
function Gaussian(μ:Real, σ:Real) -> m:Gaussian {
  m.μ <- μ;
  m.σ <- σ;
}

/**
 * Simulate.
 */
function x:Real <~ m:Gaussian -> x {
  cpp {{
  x = std::normal_distribution<double>(m.μ, m.σ)(rng);
  }}
}

/**
 * Evaluate pdf.
 */
function x:Real ~> m:Gaussian -> l:Real {
  l <- exp(log(x ~> m));
}

/**
 * Evaluate log-pdf.
 */
function log(x:Real ~> m:Gaussian) -> ll:Real {
  ll <- -0.5*pow((x - m.μ)/m.σ, 2.0) - log(sqrt(2.0*π)*m.σ);
}
