/**
 * State for SIRModel.
 */
class SIRState < State {
  /**
   * Number of susceptible-infectious interactions.
   */
  τ:Random<Integer>;

  /**
   * Newly infected population.
   */
  Δi:Random<Integer>;

  /**
   * Newly recovered population.
   */
  Δr:Random<Integer>;

  /**
   * Susceptible population.
   */
  s:Random<Integer>;

  /**
   * Infectious population.
   */
  i:Random<Integer>;

  /**
   * Recovered population.
   */
  r:Random<Integer>;

  fiber initial(θ:SIRParameter) -> Real {
    //
  }

  fiber transition(x:SIRState, θ:SIRParameter) -> Real {
    τ ~ Binomial(x.s, 1.0 - exp(-θ.λ*x.i/(x.s + x.i + x.r)));
    Δi ~ Binomial(τ, θ.δ);
    Δr ~ Binomial(x.i, θ.γ);

    s ~ Delta(x.s - Δi);
    i ~ Delta(x.i + Δi - Δr);
    r ~ Delta(x.r + Δr);
  }

  function input(reader:Reader) {
    Δi <- reader.getInteger("Δi");
    Δr <- reader.getInteger("Δr");
    s <- reader.getInteger("s");
    i <- reader.getInteger("i");
    r <- reader.getInteger("r");
  }

  function output(writer:Writer) {
    writer.setInteger("Δi", Δi);
    writer.setInteger("Δr", Δr);
    writer.setInteger("s", s);
    writer.setInteger("i", i);
    writer.setInteger("r", r);
  }
}
