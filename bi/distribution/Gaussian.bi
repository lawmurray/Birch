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

  function initialize(u:LogGaussian) {
    super.initialize(u);
  }

  function initialize(μ:Real, σ2:Real) {
    super.initialize();
    update(μ, σ2);
  }

  function update(μ:Real, σ2:Real) {
    assert σ2 >= 0.0;
    
    this.μ <- μ;
    this.σ2 <- σ2;
  }

  function doRealize() {
    if (isMissing()) {
      set(simulate_gaussian(μ, σ2));
    } else {
      setWeight(observe_gaussian(x, μ, σ2));
    }
  }

  function tildeLeft() -> Gaussian {
    simulate();
    return this;
  }
  
  function tildeRight(left:Gaussian) -> Gaussian {
    set(left.value());
    observe();
    return this;
  }
}

/**
 * Synonym for Gaussian.
 */
type Normal = Gaussian;

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:Real, σ2:Real) -> Gaussian {
  m:Gaussian;
  m.initialize(μ, σ2);
  return m;
}

/**
 * Create Gaussian distribution.
 */
function Normal(μ:Real, σ2:Real) -> Gaussian {
  return Gaussian(μ, σ2);
}
