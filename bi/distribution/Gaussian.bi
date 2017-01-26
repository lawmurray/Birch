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
function ~m:Gaussian -> x:Real {
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

/**
 * Marginalise with Gaussian prior on mean.
 */
function Gaussian(μ:(Real ~ Gaussian), σ:Real) -> m2:Gaussian {
  m1:Gaussian;
  m1 <- μ;
  m2.μ <- m1.μ;
  m2.σ <- sqrt(pow(m1.σ, 2.0) + pow(σ, 2.0));
}

/**
 * Condition with Gaussian prior on mean.
 */
function y:Real ~> Gaussian(μ:(Real ~ Gaussian), σ:Real) -> l:Real {
  /* variances and precisions */
  m:Gaussian;
  m <- μ;
  σ2_0:Real <- pow(m.σ, 2.0);
  λ_0:Real <- 1.0/σ2_0;
  σ2:Real <- pow(σ, 2.0);
  λ:Real <- 1.0/σ2;

  /* prior likelihood */
  l <- y ~> Gaussian(m.μ, sqrt(σ2_0 + σ2));

  /* posterior sample */
  σ2_1:Real <- 1.0/(λ_0 + λ);
  μ <- Gaussian((m.μ*λ_0 + y*λ)*σ2_1, sqrt(σ2_1));
}
