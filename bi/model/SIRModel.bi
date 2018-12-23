/**
 * SIR model parameters.
 */
class SIRParameter {
  /**
   * Interaction rate.
   */
  λ:Random<Real>;

  /**
   * Infection probability.
   */
  δ:Random<Real>;

  /**
   * Recovery probability.
   */
  γ:Random<Real>;

  function read(buffer:Buffer) {
    buffer.get("λ", λ);
    buffer.get("δ", δ);
    buffer.get("γ", γ);
  }

  function write(buffer:Buffer) {
    buffer.set("λ", λ);
    buffer.set("δ", δ);
    buffer.set("γ", γ);
  }
}

/**
 * SIR model state.
 */
class SIRState {
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

  function read(buffer:Buffer) {
    buffer.get("Δi", Δi);
    buffer.get("Δr", Δr);
    buffer.get("s", s);
    buffer.get("i", i);
    buffer.get("r", r);
  }

  function write(buffer:Buffer) {
    buffer.set("Δi", Δi);
    buffer.set("Δr", Δr);
    buffer.set("s", s);
    buffer.set("i", i);
    buffer.set("r", r);
  }
}

/**
 * SIR (susceptible-infectious-recovered) model for infectious disease
 * outbreaks in epidemiology.
 *
 * The model on which this is based is described in
 * [Murray et. al. (2018)](../#references).
 *
 * The parameter model is given by:
 *   $$\begin{align}
 *   \lambda &\sim \mathrm{Gamma}(2,5) \\
 *   \delta &\sim \mathrm{Beta}(2,2) \\
 *   \gamma &\sim \mathrm{Beta}(2,2),
 *   \end{align}$$
 * where $\lambda$ is a rate of interaction in the population, $\delta$ the
 * probability of infection when a susceptible individual interacts with an
 * infectious individual, and $\gamma$ the daily recovery probability.
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
class SIRModel < MarkovModel<SIRParameter,SIRState> {
  fiber parameter(θ':SIRParameter) -> Real {
    θ'.λ ~ Gamma(2.0, 5.0);
    θ'.δ ~ Beta(2.0, 2.0);
    θ'.γ ~ Beta(2.0, 2.0);
  }

  fiber initial(x:SIRState, θ:SIRParameter) -> Real {
    //
  }

  fiber transition(x':SIRState, x:SIRState, θ:SIRParameter) -> Real {
    x'.τ ~ Binomial(x.s, 1.0 - exp(-θ.λ*x.i/(x.s + x.i + x.r)));
    x'.Δi ~ Binomial(x'.τ, θ.δ);
    x'.Δr ~ Binomial(x.i, θ.γ);

    x'.s ~ Delta(x.s - x'.Δi);
    x'.i ~ Delta(x.i + x'.Δi - x'.Δr);
    x'.r ~ Delta(x.r + x'.Δr);
  }
}
