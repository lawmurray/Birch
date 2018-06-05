/**
 * SIR state model.
 *
 * The initial model for time $t = 0$ is:
 *   $$\begin{align}
 *   s_0 &= 760 \\
 *   i_0 &= 3 \\
 *   r_0 &= 0.
 *   \end{align}$$
 *
 * The transition model for time $t$ is:
 *   $$\begin{align}
 *   \tau_t &\sim \mathrm{Binomial}\left(s_{t-1}, 1 - \exp\left(\frac{-\lambda
 *   i_{t-1} }{s_{t-1} + i_{t-1} + r_{t-1}}\right) \right) \\
 *   \Delta i_t &\sim \mathrm{Binomial}(\tau_t, \delta) \\
 *   \Delta r_t &\sim \mathrm{Binomial}(i_{t-1}, \gamma),
 *   \end{align}$$
 * where $\tau_t$ is the number of interactions between infectious and
 * susceptible individuals, $\Delta i_t$ the number of newly infected
 * individuals, and $\Delta r_t$ the number of newly recovered individuals.
 *
 * Population counts are then updated:
 * $$\begin{align}
 * s_t &= s_{t-1} - \Delta i_t \\
 * i_t &= i_{t-1} + \Delta i_t - \Delta r_t \\
 * r_t &= r_{t-1} + \Delta r_t.
 * \end{align}$$
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
