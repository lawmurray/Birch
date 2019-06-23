/*
 * Delayed Gaussian random variate.
 */
class DelayGaussian(future:Real?, futureUpdate:Boolean, μ:Real, σ2:Real) <
    DelayValue<Real>(future, futureUpdate) {
  /**
   * Mean.
   */
  μ:Real <- μ;

  /**
   * Precision.
   */
  λ:Real <- 1.0/σ2;

  function simulate() -> Real {
    return simulate_gaussian(μ, 1.0/λ);
  }
  
  function logpdf(x:Real) -> Real {
    return logpdf_gaussian(x, μ, 1.0/λ);
  }
  
  function update(x:Real) {
    //
  }

  function downdate(x:Real) {
    //
  }

  function pdf(x:Real) -> Real {
    return pdf_gaussian(x, μ, 1.0/λ);
  }

  function cdf(x:Real) -> Real {
    return cdf_gaussian(x, μ, 1.0/λ);
  }

  function write(buffer:Buffer) {
    parent:Delay? <- this.parent;
    if parent? {
      /* can only output as a distribution if this is a root node on the
       * $M$-path */
      super.write(buffer);
    } else {
      prune();
      buffer.set("class", "Gaussian");
      buffer.set("μ", μ);
      buffer.set("σ2", 1.0/λ);
    }
  }
}

function DelayGaussian(future:Real?, futureUpdate:Boolean, μ:Real,
    σ2:Real) -> DelayGaussian {
  m:DelayGaussian(future, futureUpdate, μ, σ2);
  return m;
}
