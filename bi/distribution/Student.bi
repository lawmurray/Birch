/**
 * Student's $t$-distribution.
 */
class Student < Random<Real> {
  /**
   * Degrees of freedom.
   */
  ν:Real;

  function initialize(ν:Real) {
    super.initialize();
    update(ν);
  }

  function update(ν:Real) {
    assert ν > 0.0;
    
    this.ν <- ν;
  }

  function doRealize() {
    if (isMissing()) {
      set(simulate_student_t(ν));
    } else {
      setWeight(observe_student_t(value(), ν));
    }
  }
}

/**
 * Create Student's $t$-distribution.
 */
function Student(ν:Real) -> Student {
  m:Student;
  m.initialize(ν);
  return m;
}
