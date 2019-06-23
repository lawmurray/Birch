/*
 * Delayed Student's $t$ random variate.
 */
final class DelayStudent(future:Real?, futureUpdate:Boolean, ν:Real, μ:Real, σ2:Real) < DelayValue<Real>(future, futureUpdate) {
  /**
   * Degrees of freedom.
   */
  ν:Real <- ν;

  /**
   * Location parameter.
   */
  μ:Real <- μ;

  /**
   * Squared scale parameter.
   */
  σ2:Real <- σ2;
  
  function simulate() -> Real {
    return simulate_student_t(ν, μ, σ2);
  }
  
  function logpdf(x:Real) -> Real {
    return logpdf_student_t(x, ν, μ, σ2);
  }

  function update(x:Real) {
    //
  }

  function downdate(x:Real) {
    //
  }

  function pdf(x:Real) -> Real {
    return pdf_student_t(x, ν, μ, σ2);
  }

  function cdf(x:Real) -> Real {
    return cdf_student_t(x, ν, μ, σ2);
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Student");
    buffer.set("ν", ν);
    buffer.set("μ", μ);
    buffer.set("σ2", σ2);
  }
}

function DelayStudent(future:Real?, futureUpdate:Boolean, ν:Real, μ:Real, σ2:Real) -> DelayStudent {
  m:DelayStudent(future, futureUpdate, ν, μ, σ2);
  return m;
}
