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
function ~m:Gaussian -> x:Real {
  cpp {{
  x = std::normal_distribution<double>(m.μ, m.σ)(rng);
  }}
}

/**
 * Weight.
 */
function x:Real ~> m:Gaussian -> l:Real {
  l <- exp(log(x ~> m));
}

/**
 * Log-weight.
 */
function log(x:Real ~> m:Gaussian) -> ll:Real {
  ll <- -0.5*pow((x - m.μ)/m.σ, 2.0) - log(sqrt(2.0*π)*m.σ);
}

/**
 * Forward functions.
 */
function Gaussian(μ:Gaussian, σ:Real) -> m:Gaussian {
  m.μ <- μ.μ;
  m.σ <- sqrt(pow(μ.σ, 2.0) + pow(σ, 2.0));
}

/**
 * Backward functions.
 */
function x:Real ~> Gaussian(μ:Gaussian!, σ:Real) -> l:Real {
  /* variances and precisions */
  σ2_0:Real <- pow(μ.σ, 2.0);
  λ_0:Real <- 1.0/σ2_0;
  σ2:Real <- pow(σ, 2.0);
  λ:Real <- 1.0/σ2;

  /* prior likelihood */
  l <- x ~> Gaussian(μ.μ, sqrt(σ2_0 + σ2));

  /* posterior update */
  σ2_1:Real <- 1.0/(λ_0 + λ);
  μ.μ <- (μ.μ*λ_0 + x*λ)*σ2_1;
  μ.σ <- sqrt(σ2_1);
}
