/**
 * SEIR process model.
 */
class SEIRState {
  n:Integer;    // total population
  s:Binomial;   // susceptible population
  e:Binomial;   // incubating population
  i:Binomial;   // infectious population
  r:Binomial;   // recovered population

  Δs:Poisson;   // newly susceptible (births)
  Δe:Binomial;  // newly exposed
  Δi:Binomial;  // newly infected
  Δr:Binomial;  // newly recovered
  
  /**
   * Transition model.
   *
   *   - n: exchange trials.
   *   - θ: parameters.
   */
  function transition(n:SEIRExchange, θ:SEIRParameter) -> SEIRState {
    assert 0 <= n.s && n.s <= s;
    assert 0 <= n.e && n.e <= e;
    assert 0 <= n.i && n.i <= i;

    x:SEIRState;  // new state
  
    Δs ~ Poisson(θ.ν);
    Δe ~ Binomial(n.s, θ.λ);
    Δi ~ Binomial(n.e, θ.δ);
    Δr ~ Binomial(n.i, θ.γ);

    x.s ~ Binomial(s + Δs - Δe, 1.0 - θ.μ);
    x.e ~ Binomial(e + Δe - Δi, 1.0 - θ.μ);
    x.i ~ Binomial(i + Δi - Δr, 1.0 - θ.μ);
    x.r ~ Binomial(r + Δr, 1.0 - θ.μ);
    x.n <- x.s + x.e + x.i + x.r;

    return x;
  }
}
