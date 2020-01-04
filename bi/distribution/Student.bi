/**
 * Student's $t$-distribution.
 */
final class Student(future:Real?, futureUpdate:Boolean,
    ν:Expression<Real>, μ:Expression<Real>, σ2:Expression<Real>) <
    Distribution<Real>(future, futureUpdate) {
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
    return simulate_student_t(ν, μ, σ2);
  }
  
  function logpdf(x:Real) -> Real {
    return logpdf_student_t(x, ν, μ, σ2);
  }

  function cdf(x:Real) -> Real? {
    return cdf_student_t(x, ν, μ, σ2);
  }

  function quantile(p:Real) -> Real? {
    return quantile_student_t(p, ν, μ, σ2);
  }

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      delay <- Student(future, futureUpdate, ν, μ, σ2);
    }
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Student");
    buffer.set("ν", ν);
    buffer.set("μ", μ);
    buffer.set("σ2", σ2);
  }
}

function Student(future:Real?, futureUpdate:Boolean, ν:Real, μ:Real, σ2:Real) -> Student {
  m:Student(future, futureUpdate, ν, μ, σ2);
  return m;
}

/**
 * Create Student's $t$-distribution.
 */
function Student(ν:Expression<Real>, μ:Expression<Real>, σ2:Expression<Real>) -> Student {
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
