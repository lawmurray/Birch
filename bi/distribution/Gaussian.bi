import distribution.DelayReal;
import math;
import random;

/**
 * Gaussian distribution.
 */
class Gaussian < DelayReal {
  /**
   * Mean.
   */
  μ:Real;
  
  /**
   * Variance.
   */
  σ2:Real;

  function initialize(u:Gaussian) {
    super.initialize(u);
  }

  function initialize(μ:Real, σ2:Real) {
    assert σ2 > 0.0;
    
    super.initialize();
    this.μ <- μ;
    this.σ2 <- σ2;
  }

  function update(μ:Real, σ2:Real) {
    assert σ2 > 0.0;
    
    this.μ <- μ;
    this.σ2 <- σ2;
  }

  function doSimulate() {
    cpp {{
    set_(std::normal_distribution<double>(μ_, ::sqrt(σ2_))(rng));
    }}    
  }
  
  function doObserve() {
    setWeight(-0.5*(pow((x - μ), 2.0)/σ2 - log(σ2) - log(2.0*π)));
  }
}

/**
 * Create.
 */
function Gaussian(μ:Real, σ2:Real) -> Gaussian {
  m:Gaussian;
  m.initialize(μ, σ2);
  return m;
}
