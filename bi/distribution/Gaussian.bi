import math;
import random;

/**
 * Gaussian distribution.
 */
model Gaussian(μ1:Real, σ1:Real) {
  /**
   * Mean.
   */
  μ:Real <- μ1;
  
  /**
   * Standard deviation.
   */
  σ:Real <- σ1;
}

/**
 * Simulate.
 */
function m:Gaussian ~> x:Real {
  cpp {{
  x = std::normal_distribution<double>(m.μ, m.σ)(rng);
  }}
}

/**
 * Evaluate pdf.
 */
function x:Real ~ m:Gaussian -> l:Real {
  l <- exp(log(x ~ m));
}

/**
 * Evaluate log-pdf.
 */
function log(x:Real ~ m:Gaussian) -> ll:Real {
  ll <- -0.5*pow((x - m.μ)/m.σ, 2.0) - log(sqrt(2.0*π)*m.σ);
}
