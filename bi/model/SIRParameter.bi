/**
 * SIR parameter model.
 *
 * The model is given by:
 *   $$\begin{align}
 *   \lambda &= 10 \\
 *   \delta &\sim \mathrm{Beta}(2,2) \\
 *   \gamma &\sim \mathrm{Beta}(2,2),
 *   \end{align}$$
 * where $\lambda$ is a rate of interaction in the population, $\delta$ the
 * probability of infection when a susceptible individual interacts with an
 * infectious individual, and $\gamma$ the daily recovery probability.
 */
class SIRParameter < Parameter {
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

  fiber parameter() -> Real {
    λ <- 10.0;
    δ ~ Beta(2.0, 2.0);
    γ ~ Beta(2.0, 2.0);
  }

  function input(reader:Reader) {
    λ <- reader.getReal("λ");
    δ <- reader.getReal("δ");
    γ <- reader.getReal("γ");
  }

  function output(writer:Writer) {
    writer.setReal("λ", λ);
    writer.setReal("δ", δ);
    writer.setReal("γ", γ);
  }
}
