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
 * components of the joint distribution, rather than the `simulate` member
 * fiber..
 *
 * In addition to the usual fiber-based interface for models, `MarkovModel`
 * provides an alternative function-based interface based on the `start` and
 * `stop` member functions.
 */
class MarkovModel<Parameter,State> < StateModel<Parameter,State> {
  fiber simulate() -> Real {
    /* parameters */
    yield sum(parameter(θ));
    
    /* states */
    auto f <- this.x.walk();    
    u:State?;  // previous state
    x:State?;  // current state
    while true {
      if f? {  // is the next state given?
        x <- f!;
      } else {
        o:State;
        this.x.pushBack(o);
        x <- o;
      }
      if u? {
        yield sum(transition(x!, u!, θ));
      } else {
        yield sum(initial(x!, θ));
      }
      u <- x;
    }
  }
  
  /**
   * Start simulation of the model, using the incremental interface.
   *
   * Returns: log-weight from the parameter model.
   */
  function start() -> Real {
    return sum(parameter(θ));
  }

  /**
   * Continue simulation of the model one step, using the incremental
   * interface.
   *
   * Returns: log-weight of the initial or transition model.
   */
  function step() -> Real {
    w:Real <- 0.0;    
    x:State;
    if this.x.empty() {
      w <- sum(initial(x, θ));
    } else {
      w <- sum(transition(x, this.x.back(), θ));
    }
    this.x.pushBack(x);
    return w;
  }

  function checkpoints() -> Integer? {
    /* one checkpoint for the parameters, then one for each time */
    return 1 + x.size();
  }
  
  /**
   * Initial model.
   *
   * - x: The initial state, to be set.
   * - θ: The parameters.
   */
  fiber initial(x:State, θ:Parameter) -> Real {
    //
  }
  
  /**
   * Transition model.
   *
   * - x: The current state, to be set.
   * - u: The previous state.
   * - θ: The parameters.
   */
  fiber transition(x:State, u:State, θ:Parameter) -> Real {
    //
  }
}
