/*
 * Delayed Student's $t$ random variate.
 */
class DelayStudent(x:Random<Real>, ν:Real) < DelayValue<Real>(x) {
  /**
   * Degrees of freedom.
   */
  ν:Real <- ν;
  
  function doSimulate() -> Real {
    return simulate_student_t(ν);
  }
  
  function doObserve(x:Real) -> Real {
    return observe_student_t(x, ν);
  }

  function pdf(x:Real) -> Real {
    return pdf_student_t(x, ν);
  }

  function cdf(x:Real) -> Real {
    return cdf_student_t(x, ν);
  }
}

function DelayStudent(x:Random<Real>, ν:Real) -> DelayStudent {
  m:DelayStudent(x, ν);
  return m;
}
