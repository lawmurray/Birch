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
class MarkovModel<Parameter,State> < ForwardModel {
  /**
   * Parameter.
   */
  θ:Parameter;

  /**
   * States.
   */
  x:Queue<State>;
  
  /**
   * Current state during simulation.
   */
  f:QueueNode<State>?;

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
  function start() -> Real {
    return h.handle(parameter(θ));
  }

  /**
   * Play one step. Simulates through the next state.
   */
  function play() -> Real {
    x:State?;
    if f? {
      x <- f!.getValue();
      f <- f!.getNext();  
    } else {
      f <- this.x.begin();
    }
    if !f? {
      /* no next state, insert one */
      x':State;
      this.x.pushBack(x');
      f <- this.x.end();
    }

    auto x' <- f!.getValue();
    auto w' <- 0.0;
    if !x? {
      w' <- w' + h.handle(initial(x', θ));
    } else {
      w' <- w' + h.handle(transition(x', x!, θ));
    }
    return w';
  }

  function size() -> Integer {
    return x.size();
  }
  
  function rewind() {
    f <- nil;
  }

  fiber simulate() -> Event {
    start();
    while true {
      play();
    }
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
