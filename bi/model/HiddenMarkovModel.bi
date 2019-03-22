/**
 * Hidden Markov Model (HMM) or state-space model (SSM).
 *
 * The joint distribution is:
 *
 * $$p(\mathrm{d}\theta, \mathrm{d}x_{0:T}, \mathrm{d}y_{0:T}) =
 *   p(\mathrm{d}\theta) p(\mathrm{d}x_0 \mid \theta)  p(\mathrm{d}y_0
 *   \mid x_0, \theta) \prod_{t=1}^T p(\mathrm{d}x_t \mid x_{t-1},
 *   \theta) p(\mathrm{d}y_t \mid x_t, \theta)$$
 *
 * Typically the $Y_{0:T}$ are observed, although they need not be.
 *
 * <center>
 * ![Graphical model depicting `HiddenMarkovModel`/`StateSpaceModel`.](../figs/HiddenMarkovModel.svg)
 * </center>
 *
 * A model inheriting from `HiddenMarkovModel`/`StateSpaceModel` overrides
 * the `parameter`, `initial`, `transition` and `observation` member fibers
 * to specify the individual components of the joint distribution.
 */
class HiddenMarkovModel<Parameter,State,Observation> <
    MarkovModel<Parameter,State> {
  /**
   * Observations.
   */
  y:List<Observation>;

  /**
   * Current observation during simulation.
   */
  g:ListNode<Observation>?;

  /**
   * Observation model.
   *
   * - y: The observations, to be set.
   * - x: The current state.
   * - θ: The parameters.
   */
  fiber observation(y:Observation, x:State, θ:Parameter) -> Event {
    //
  }

  /**
   * Play one step. Simulates through the next state and observation.
   */
  fiber play() -> Event {
    super.play();
    if g? {
      /* move to next observation */
      g <- g!.getNext();
    }
    if !g? {
      /* if there is no next observation, insert one */
      y:Observation;
      this.y.pushBack(y);
      g <- this.y.end();
    }
    auto x <- f.getValue();
    auto y <- g.getValue();
    observation(y, x, θ);
  }

  function checkpoints() -> Integer? {
    return max(x.size(), y.size());
  }

  function read(buffer:Buffer) {
    super.read(buffer);
    buffer.get("y", y);
  }
  
  function write(buffer:Buffer) {
    super.write(buffer);
    buffer.set("y", y);
  }
}
