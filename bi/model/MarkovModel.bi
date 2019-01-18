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
 * fiber; likewise for the `propose` analogs.
 *
 * In addition to the usual fiber-based interface for models, `MarkovModel`
 * provides an alternative function-based interface based on the `start` and
 * `stop` member functions.
 */
class MarkovModel<Parameter,State> < Model {
  /**
   * Parameter.
   */
  θ:Parameter;

  /**
   * States.
   */
  x:List<State>;

  fiber simulate() -> Real {
    auto f <- this.x.walk();
    
    u:State?;  // previous state
    x:State?;  // current state

    yield sum(parameter(θ));
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

  fiber propose() -> Real {
    auto f <- this.x.walk();
    
    u:State?;  // previous state
    x:State?;  // current state

    yield sum(proposeParameter(θ));
    while true {
      if f? {  // is the next state given?
        x <- f!;
      } else {
        o:State;
        this.x.pushBack(o);
        x <- o;
      }
      if u? {
        yield sum(proposeTransition(x!, u!, θ));
      } else {
        yield sum(proposeInitial(x!, θ));
      }
      u <- x;
    }
  }

  fiber propose(m:Model) -> Real {
    auto n <- MarkovModel<Parameter,State>?(m);
    if n? {
      propose(n!);
    } else {
      error("previous state has incorrect type");
    }
  }
  
  fiber propose(m:MarkovModel<Parameter,State>) -> Real {  
    auto θ <- m.θ;
    auto θ' <- this.θ;
    auto f <- m.x.walk(); 
    auto f' <- this.x.walk();
    
    u:State?;  // previous state of m
    x:State?;  // current state of m
    u':State?;  // previous state of this
    x':State?;  // current state of this

    yield sum(proposeParameter(θ', θ));
    while true {
      if f? {
        x <- f!;
      } else {
        error("previous state has incorrect number of checkpoints");
      }
      if f'? {  // is the next state given?
        x' <- f'!;
      } else {
        o:State;
        this.x.pushBack(o);
        x' <- o;
      }
      if u'? {
        yield sum(proposeTransition(x'!, u'!, θ', x!, u!, θ));
      } else {
        yield sum(proposeInitial(x'!, θ', x!, θ));
      }
      u <- x;
      u' <- x';
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
   * Parameter model.
   *
   * - θ: The parameters, to be set.
   */
  fiber parameter(θ:Parameter) -> Real {
    //
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

  /**
   * Parameter proposal.
   *
   * - θ: The parameters, to be set.
   *
   * By default calls `parameter(θ)`.
   */
  fiber proposeParameter(θ:Parameter) -> Real {
    parameter(θ);
  }
  
  /**
   * Initial proposal.
   *
   * - x: The initial state, to be set.
   * - θ: The parameters.
   *
   * By default calls `initial(x, θ)`.
   */
  fiber proposeInitial(x:State, θ:Parameter) -> Real {
    initial(x, θ);
  }
  
  /**
   * Transition proposal.
   *
   * - x: The current state, to be set.
   * - u: The previous state.
   * - θ: The parameters.
   *
   * By default calls `transition(x, u, θ)`.
   */
  fiber proposeTransition(x:State, u:State, θ:Parameter) -> Real {
    transition(x, u, θ);
  }

  /**
   * Parameter proposal.
   *
   * - θ': The proposed parameters, to be set.
   * - θ: The last parameters.
   *
   * By default calls `proposeParameter(θ')`.
   */
  fiber proposeParameter(θ':Parameter, θ:Parameter) -> Real {
    proposeParameter(θ');
  }
  
  /**
   * Initial proposal.
   *
   * - x': The initial state, to be set.
   * - θ': The parameters.
   * - x: The last initial state.
   * - θ: The last parameters.
   *
   * By default calls `proposeInitial(x', θ')`.
   */
  fiber proposeInitial(x':State, θ':Parameter, x:State, θ:Parameter) -> Real {
    proposeInitial(x', θ');
  }
  
  /**
   * Transition proposal.
   *
   * - x': The current state, to be set.
   * - u': The previous state.
   * - θ': The parameters.
   * - x: The last current state.
   * - u: The last previous state.
   * - θ: The last parameters.
   *
   * By default calls `proposeTransition(x', u', θ')`.
   */
  fiber proposeTransition(x':State, u':State, θ':Parameter, x:State, u:State,
      θ:Parameter) -> Real {
    proposeTransition(x', u', θ');
  }

  function read(buffer:Buffer) {
    buffer.get("θ", θ);
    buffer.get("x", x);
  }
  
  function write(buffer:Buffer) {
    buffer.set("θ", θ);
    buffer.set("x", x);
  }
}
