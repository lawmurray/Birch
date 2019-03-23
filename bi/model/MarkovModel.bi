/**
 * Markov model.
 *
 * The joint distribution is:
 *
 * $$p(\mathrm{d}\theta, \mathrm{d}x_{0:T}) =
 *   p(\mathrm{d}\theta) p(\mathrm{d}x_0 \mid \theta) 
 *   \prod_{t=1}^T p(\mathrm{d}x_t \mid x_{t-1}, \theta).$$
 *
 * <center>
 * ![Graphical model depicting MarkovModel.](../figs/MarkovModel.svg)
 * </center>
 *
 * A model inheriting from `MarkovModel` overrides the `parameter`,
 * `initial` and `transition` member fibers to specify the individual
 * components of the joint distribution.
 */
class MarkovModel<Parameter,State> < BidirectionalModel {
  /**
   * Parameter.
   */
  θ:Parameter;

  /**
   * States.
   */
  x:List<State>;
  
  /**
   * Current state during simulation.
   */
  f:ListNode<State>?;

  /**
   * Parameter model.
   *
   * - θ: The parameters, to be set.
   */
  fiber parameter(θ:Parameter) -> Event {
    //
  }
  
  /**
   * Initial model.
   *
   * - x: The initial state, to be set.
   * - θ: The parameters.
   */
  fiber initial(x:State, θ:Parameter) -> Event {
    //
  }
  
  /**
   * Transition model.
   *
   * - x: The current state, to be set.
   * - u: The previous state.
   * - θ: The parameters.
   */
  fiber transition(x:State, u:State, θ:Parameter) -> Event {
    //
  }
  
  /**
   * Start. Simulates through the parameter model.
   */
  fiber start() -> Event {
    parameter(θ);
  }

  /**
   * Play one step. Simulates through the next state.
   */
  fiber step() -> Event {
    if f? {
      /* move to next state */
      f <- f!.getNext();
    }
    if !f? {
      /* if there is no next state, insert one */
      x:State;
      this.x.pushBack(x);
      f <- this.x.begin();
    }
    auto x <- f.getValue();
    if x == f.begin() {
      initial(x, θ);
    } else {
      auto u <- f.getPrev()!.getValue();
      transition(x, u, θ);
    }
  }

  fiber simulate() -> Event {
    start();
    while true {
      step();
    }
  }

  function size() -> Integer? {
    return x.size();
  }

  function read(buffer:Buffer) {
    super.read(buffer);
    buffer.get("θ", θ);
    buffer.get("x", x);
  }
  
  function write(buffer:Buffer) {
    super.write(buffer);
    buffer.set("θ", θ);
    buffer.set("x", x);
  }
}
