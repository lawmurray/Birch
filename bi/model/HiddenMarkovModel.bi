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
  y:Queue<Observation>;

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
  function play() -> Real {
    auto w <- super.play();
    auto x <- f!.getValue();
    auto y <- g!.getValue();
    return w + h.handle(observation(y, x, θ));
  }

  function size() -> Integer {
    return max(x.size(), y.size());
  }

  function next() {
    super.next();
    if g? {
      g <- g!.getNext();
    } else {
      g <- y.begin();
    }    
    if !g? {
      /* no next observation, insert one */
      y':Observation;
      y.pushBack(y');
      g <- y.end();
    }
  }

  function rewind() {
    super.rewind();
    g <- nil;
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
