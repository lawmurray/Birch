/*
 * Delayed Student's $t$ random variate.
 */
final class DelayStudent(future:Real?, futureUpdate:Boolean, ν:Real) <
    DelayValue<Real>(future, futureUpdate) {
  /**
   * Degrees of freedom.
   */
  ν:Real <- ν;
  
  function simulate() -> Real {
    return simulate_student_t(ν);
  }
  
  function observe(x:Real) -> Real {
    return observe_student_t(x, ν);
  }

  function update(x:Real) {
    //
  }

  function downdate(x:Real) {
    //
  }

  function pdf(x:Real) -> Real {
    return pdf_student_t(x, ν);
  }

  function cdf(x:Real) -> Real {
    return cdf_student_t(x, ν);
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Student");
    buffer.set("ν", ν);
  }
}

function DelayStudent(future:Real?, futureUpdate:Boolean, ν:Real) ->
    DelayStudent {
  m:DelayStudent(future, futureUpdate, ν);
  return m;
}
