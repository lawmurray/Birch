/*
 * Delayed Chinese restaurant process (CRP) random variate. Such a variate
 * cannot be instantiated, but the associated random variable may be
 * marginalized out.
 */
final class DelayRestaurant(future:Real[_]?, futureUpdate:Boolean, α:Real,
    θ:Real) < DelayValue<Real[_]>(future, futureUpdate) {
  /**
   * Concentration.
   */
  α:Real <- α;
  
  /**
   * Strength.
   */
  θ:Real <- θ;

  /**
   * Number of samples drawn in each component.
   */
  n:Integer[_];

  /**
   * Number of components enumerated.
   */
  K:Integer <- 0;

  /**
   * Number of samples drawn.
   */
  N:Integer <- 0;

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Restaurant");
    buffer.set("α", α);
    buffer.set("θ", θ);
    buffer.set("n", n);
  }
}

function DelayRestaurant(future:Real[_]?, futureUpdate:Boolean, α:Real,
    θ:Real) -> DelayRestaurant {
  m:DelayRestaurant(future, futureUpdate, α, θ);
  return m;
}
