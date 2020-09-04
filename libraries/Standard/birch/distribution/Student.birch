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
  
  function supportsLazy() -> Boolean {
    return true;
  }

  function simulate() -> Real {
    return simulate_student_t(ν.value(), μ.value(), σ2.value());
  }

  function simulateLazy() -> Real? {
    return simulate_student_t(ν.get(), μ.get(), σ2.get());
  }
  
  function logpdf(x:Real) -> Real {
    return logpdf_student_t(x, ν.value(), μ.value(), σ2.value());
  }

  function logpdfLazy(x:Expression<Real>) -> Expression<Real>? {
    return logpdf_lazy_student_t(x, ν, μ, σ2);
  }

  function cdf(x:Real) -> Real? {
    return cdf_student_t(x, ν.value(), μ.value(), σ2.value());
  }

  function quantile(P:Real) -> Real? {
    return quantile_student_t(P, ν.value(), μ.value(), σ2.value());
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
  return construct<Student>(ν, μ, σ2);
}

/**
 * Create Student's $t$-distribution.
 */
function Student(ν:Expression<Real>) -> Student {
  return Student(ν, box(0.0), box(1.0));
}

/**
 * Create Student's $t$-distribution.
 */
function Student(ν:Real) -> Student {
  return Student(box(ν), box(0.0), box(1.0));
}

/**
 * Create Student's $t$-distribution.
 */
function Student(ν:Real, μ:Expression<Real>, σ2:Expression<Real> ) -> Student {
  return Student(box(ν), μ, σ2);
}

/**
 * Create Student's $t$-distribution.
 */
function Student(ν:Expression<Real>, μ:Real, σ2:Expression<Real> ) -> Student {
  return Student(ν, box(μ), σ2);
}

/**
 * Create Student's $t$-distribution.
 */
function Student(ν:Expression<Real>, μ:Expression<Real>, σ2:Real ) -> Student {
  return Student(ν, μ, box(σ2));
}

/**
 * Create Student's $t$-distribution.
 */
function Student(ν:Real, μ:Real, σ2:Expression<Real> ) -> Student {
  return Student(box(ν), box(μ), σ2);
}

/**
 * Create Student's $t$-distribution.
 */
function Student(ν:Real, μ:Expression<Real>, σ2:Real ) -> Student {
  return Student(box(ν), μ, box(σ2));
}

/**
 * Create Student's $t$-distribution.
 */
function Student(ν:Expression<Real>, μ:Real, σ2:Real ) -> Student {
  return Student(ν, box(μ), box(σ2));
}

/**
 * Create Student's $t$-distribution.
 */
function Student(ν:Real, μ:Real, σ2:Real ) -> Student {
  return Student(box(ν), box(μ), box(σ2));
}
