/**
 * Student's $t$-distribution.
 */
class Student(ν:Expression<Real>) < Random<Real> {
  /**
   * Degrees of freedom.
   */
  ν:Expression<Real> <- ν;

  function doSimulate() -> Real {
    return simulate_student_t(ν.value());
  }
  
  function doObserve(x:Real) -> Real {
    return observe_student_t(x, ν.value());
  }
}

/**
 * Create Student's $t$-distribution.
 */
function Student(ν:Expression<Real>) -> Student {
  m:Student(ν);
  return m;
}

/**
 * Create Student's $t$-distribution.
 */
function Student(ν:Real) -> Student {
  return Student(Literal(ν));
}
