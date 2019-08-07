/**
 * Student's $t$-distribution.
 */
final class Student(ν:Expression<Real>, μ:Expression<Real>, σ2:Expression<Real> ) < Distribution<Real> {
  /**
   * Degrees of freedom.
   */
  ν:Expression<Real> <- ν;

  /**
   * Location parameter.
   */
  μ:Expression<Real> <- μ;

  /**
   * Square Scale parameter.
   */
  σ2:Expression<Real> <- σ2;

  function valueForward() -> Real {
    assert !delay?;
    return simulate_student_t(ν, μ, σ2);
  }

  function observeForward(x:Real) -> Real {
    assert !delay?;
    return logpdf_student_t(x, ν, μ, σ2);
  }

  function graft(force:Boolean) {
    if delay? {
      delay!.prune();
    } else if force {
      delay <- DelayStudent(future, futureUpdate, ν, μ, σ2);
    }
  }
}

/**
 * Create Student's $t$-distribution.
 */
function Student(ν:Expression<Real>) -> Student {
  m:Student(ν, Boxed(0.0), Boxed(1.0));
  return m;
}

/**
 * Create Student's $t$-distribution.
 */
function Student(ν:Real) -> Student {
  return Student(Boxed(ν), Boxed(0.0), Boxed(1.0));
}

/**
 * Create Student's $t$-distribution.
 */
function Student(ν:Expression<Real>, μ:Expression<Real>, σ2:Expression<Real> ) -> Student {
  m:Student(ν, μ, σ2);
  return m;
}

/**
 * Create Student's $t$-distribution.
 */
function Student(ν:Real, μ:Expression<Real>, σ2:Expression<Real> ) -> Student {
  m:Student(Boxed(ν), μ, σ2);
  return m;
}

/**
 * Create Student's $t$-distribution.
 */
function Student(ν:Expression<Real>, μ:Real, σ2:Expression<Real> ) -> Student {
  m:Student(ν, Boxed(μ), σ2);
  return m;
}

/**
 * Create Student's $t$-distribution.
 */
function Student(ν:Expression<Real>, μ:Expression<Real>, σ2:Real ) -> Student {
  m:Student(ν, μ, Boxed(σ2));
  return m;
}

/**
 * Create Student's $t$-distribution.
 */
function Student(ν:Real, μ:Real, σ2:Expression<Real> ) ->
Student { m:Student(Boxed(ν), Boxed(μ), σ2);
  return m;
}

/**
 * Create Student's $t$-distribution.
 */
function Student(ν:Real, μ:Expression<Real>, σ2:Real ) -> Student {
  m:Student(Boxed(ν), μ, Boxed(σ2));
  return m;
}

/**
 * Create Student's $t$-distribution.
 */
function Student(ν:Expression<Real>, μ:Real, σ2:Real ) -> Student {
  m:Student(ν, Boxed(μ), Boxed(σ2));
  return m;
}

/**
 * Create Student's $t$-distribution.
 */
function Student(ν:Real, μ:Real, σ2:Real ) -> Student {
  m:Student(Boxed(ν), Boxed(μ), Boxed(σ2));
  return m;
}
