/**
 * Student's $t$ random variable with delayed sampling.
 */
class DelayStudent(x:Random<Real>, ν:Real) < DelayValue<Real>(x) {
  /**
   * Degrees of freedom.
   */
  ν:Expression<Real> <- ν;
  
  function doSimulate() -> Real {
    return simulate_student_t(ν);
  }
  
  function doObserve(x:Real) -> Real {
    return observe_student_t(x, ν);
  }
}

function DelayStudent(x:Random<Real>, ν:Real) -> DelayStudent {
  m:DelayStudent(x, ν);
  return m;
}
