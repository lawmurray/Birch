/**
 * Student's $t$-distribution.
 */
final class Student(ν:Expression<Real>, μ:Expression<Real>,
    σ2:Expression<Real>) < Distribution<Real> {
  /**
   * Degrees of freedom.
   */
  ν:Expression<Real> <- ν;

  /**
   * Location parameter.
   */
  μ:Expression<Real> <- μ;

  /**
   * Square scale parameter.
   */
  σ2:Expression<Real> <- σ2;
  
  function simulate() -> Real {
    return simulate_student_t(ν.value(), μ.value(), σ2.value());
  }
  
  function logpdf(x:Real) -> Real {
    return logpdf_student_t(x, ν.value(), μ.value(), σ2.value());
  }

  function cdf(x:Real) -> Real? {
    return cdf_student_t(x, ν.value(), μ.value(), σ2.value());
  }

  function quantile(P:Real) -> Real? {
    return quantile_student_t(P, ν.value(), μ.value(), σ2.value());
  }

  function graft() -> Distribution<Real> {
    prune();
    return this;
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Student");
    buffer.set("ν", ν);
    buffer.set("μ", μ);
    buffer.set("σ2", σ2);
  }
}

/**
 * Create Student's $t$-distribution.
 */
function Student(ν:Expression<Real>, μ:Expression<Real>,
    σ2:Expression<Real>) -> Student {
  m:Student(ν, μ, σ2);
  return m;
}

/**
 * Create Student's $t$-distribution.
 */
function Student(ν:Expression<Real>) -> Student {
  return Student(ν, Boxed(0.0), Boxed(1.0));
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
function Student(ν:Real, μ:Expression<Real>, σ2:Expression<Real> ) -> Student {
  return Student(Boxed(ν), μ, σ2);
}

/**
 * Create Student's $t$-distribution.
 */
function Student(ν:Expression<Real>, μ:Real, σ2:Expression<Real> ) -> Student {
  return Student(ν, Boxed(μ), σ2);
}

/**
 * Create Student's $t$-distribution.
 */
function Student(ν:Expression<Real>, μ:Expression<Real>, σ2:Real ) -> Student {
  return Student(ν, μ, Boxed(σ2));
}

/**
 * Create Student's $t$-distribution.
 */
function Student(ν:Real, μ:Real, σ2:Expression<Real> ) -> Student {
  return Student(Boxed(ν), Boxed(μ), σ2);
}

/**
 * Create Student's $t$-distribution.
 */
function Student(ν:Real, μ:Expression<Real>, σ2:Real ) -> Student {
  return Student(Boxed(ν), μ, Boxed(σ2));
}

/**
 * Create Student's $t$-distribution.
 */
function Student(ν:Expression<Real>, μ:Real, σ2:Real ) -> Student {
  return Student(ν, Boxed(μ), Boxed(σ2));
}

/**
 * Create Student's $t$-distribution.
 */
function Student(ν:Real, μ:Real, σ2:Real ) -> Student {
  return Student(Boxed(ν), Boxed(μ), Boxed(σ2));
}
