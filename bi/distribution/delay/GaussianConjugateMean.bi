import distribution.Gaussian;

/**
 * Gaussian distribution with conjugate prior over an unknown mean and known
 * standard deviation.
 */
class GaussianConjugateMean < Gaussian {
  /**
   * Prior over unknown mean.
   */
  u:Gaussian;

  /**
   * Value of known standard deviation.
   */
  τ:Real;
   
  function create(u:Gaussian, τ:Real) {
    initialise(u);
    this.u <- u;
    this.τ <- τ;
  }

  function doMarginalise() {
    if (u.isRealised()) {
      this.μ <- u.x;
      this.σ <- τ;
    } else {
      this.μ <- u.μ;
      this.σ <- sqrt(pow(u.σ, 2.0) + pow(τ, 2.0));
    }
  }
  
  function doRealise() {
    σ2_0:Real <- pow(u.σ, 2.0);
    λ_0:Real <- 1.0/σ2_0;
    σ2:Real <- pow(τ, 2.0);
    λ:Real <- 1.0/σ2;
    σ2_1:Real <- 1.0/(λ_0 + λ);
    μ_1:Real <- (u.μ*λ_0 + x*λ)*σ2_1;
    σ_1:Real <- sqrt(σ2_1);
    
    u.update(μ_1, σ_1);
  }
}

function Gaussian(μ:Gaussian, σ:Real) -> Gaussian {
  v:GaussianConjugateMean;
  v.create(μ, σ);
  return v;
}
