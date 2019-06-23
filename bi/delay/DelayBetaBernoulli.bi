/*
 * Delayed Beta-bernoulli random variate.
 */
final class DelayBetaBernoulli(future:Boolean?, futureUpdate:Boolean, ρ:DelayBeta) <
    DelayValue<Boolean>(future, futureUpdate) {
  /**
   * Success probability.
   */
  ρ:DelayBeta& <- ρ;

  function simulate() -> Boolean {
    return simulate_beta_bernoulli(ρ!.α, ρ!.β);
  }
  
  function logpdf(x:Boolean) -> Real {
    return logpdf_beta_bernoulli(x, ρ!.α, ρ!.β);
  }

  function update(x:Boolean) {
    (ρ!.α, ρ!.β) <- update_beta_bernoulli(x, ρ!.α, ρ!.β);
  }

  function downdate(x:Boolean) {
    (ρ!.α, ρ!.β) <- downdate_beta_bernoulli(x, ρ!.α, ρ!.β);
  }

  function pdf(x:Boolean) -> Real {
    return pdf_beta_bernoulli(x, ρ!.α, ρ!.β);
  }

  function write(buffer:Buffer) {
    buffer.set(value());
  }
}

function DelayBetaBernoulli(future:Boolean?, futureUpdate:Boolean, ρ:DelayBeta) ->
    DelayBetaBernoulli {
  m:DelayBetaBernoulli(future, futureUpdate, ρ);
  ρ.setChild(m);
  return m;
}
