/*
 * Delayed Chinese restaurant process (CRP) random variate. Such a variate
 * cannot be instantiated, but the associated random variable may be
 * marginalized out.
 */
class DelayRestaurant(x:Random<Real[_]>&, α:Real, θ:Real) <
    DelayValue<Real[_]>(x) {
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
}

function DelayRestaurant(x:Random<Real[_]>&, α:Real, θ:Real) ->
    DelayRestaurant {
  m:DelayRestaurant(x, α, θ);
  return m;
}
