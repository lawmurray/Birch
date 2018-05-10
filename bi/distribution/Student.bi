/**
 * Student's $t$-distribution.
 */
class Student(ν:Expression<Real>) < Distribution<Real> {
  /**
   * Degrees of freedom.
   */
  ν:Expression<Real> <- ν;

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayStudent(x, ν);
    }
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
