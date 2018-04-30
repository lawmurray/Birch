/**
 * Student's $t$-distribution.
 */
class Student(ν:Expression<Real>) < Random<Real> {
  /**
   * Degrees of freedom.
   */
  ν:Expression<Real> <- ν;

  function doGraft() -> DelayValue<Real>? {
    return DelayStudent(this, ν);
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
  return Student(Boxed(ν));
}
