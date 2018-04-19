/**
 * Student's $t$-distribution.
 */
class Student<Type1>(ν:Type1) < Random<Real> {
  /**
   * Degrees of freedom.
   */
  ν:Type1 <- ν;

  function update(ν:Type1) {
    this.ν <- ν;
  }

  function doSimulate() -> Real {
    return simulate_student_t(ν);
  }
  
  function doObserve(x:Real) -> Real {
    return observe_student_t(x, ν);
  }
}

/**
 * Create Student's $t$-distribution.
 */
function Student(ν:Real) -> Student<Real> {
  m:Student<Real>(ν);
  m.initialize();
  return m;
}

/**
 * Create Student's $t$-distribution.
 */
function Student(ν:Expression<Real>) -> Student<Expression<Real>> {
  m:Student<Expression<Real>>(ν);
  m.initialize();
  return m;
}
