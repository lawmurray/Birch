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
 * to specify the individual components of the joint distribution, rather
 * than the `simulate` member fiber; likewise for the `propose` analogs.
 *
 * As `MarkovModel`, from which they inherit,
 * `HiddenMarkovModel`/`StateSpaceModel` provides an alternative
 * function-based interface based on the `start` and `stop` member functions.
 */
class HiddenMarkovModel<Parameter,State,Observation> <
    ObservedModel<Parameter,State,Observation> {
  fiber simulate() -> Real {
    /* parameters */
    yield sum(parameter(θ));
    
    /* states and observations */
    auto f <- this.x.walk();
    auto g <- this.y.walk();
    
    u:State?;        // previous state
    x:State?;        // current state
    y:Observation?;  // current observation
    
    while true {
      w:Real <- 0.0;
      if f? {  // is the next state given?
        x <- f!;
      } else {
        o:State;
        this.x.pushBack(o);
        x <- o;
      }
      if u? {
        w <- sum(transition(x!, u!, θ));
      } else {
        w <- sum(initial(x!, θ));
      }
      u <- x;
      
      if g? {  // is the next observation given?
        y <- g!;
      } else {
        o:Observation;
        this.y.pushBack(o);
        y <- o;
      }
      w <- w + sum(observation(y!, x!, θ));
      yield w;
    }
  }

  fiber propose() -> Real {
    auto f <- this.x.walk();
    auto g <- this.y.walk();
    
    u:State?;        // previous state
    x:State?;        // current state
    y:Observation?;  // current observation

    yield sum(proposeParameter(θ));
    while true {
      w:Real <- 0.0;
      if f? {  // is the next state given?
        x <- f!;
      } else {
        o:State;
        this.x.pushBack(o);
        x <- o;
      }
      if u? {
        w <- sum(proposeTransition(x!, u!, θ));
      } else {
        w <- sum(proposeInitial(x!, θ));
      }
      u <- x;
      
      if g? {  // is the next observation given?
        y <- g!;
      } else {
        o:Observation;
        this.y.pushBack(o);
        y <- o;
      }
      w <- w + sum(proposeObservation(y!, x!, θ));
      yield w;
    }
  }

  fiber propose(m:Model) -> Real {
    auto n <- StateSpaceModel<Parameter,State,Observation>?(m);
    if n? {
      propose(n!);
    } else {
      error("previous state has incorrect type");
    }
  }
  
  fiber propose(m:StateSpaceModel<Parameter,State,Observation>) -> Real {  
    auto θ <- m.θ;
    auto θ' <- this.θ;
    auto f <- m.x.walk(); 
    auto f' <- this.x.walk();
    auto g <- m.y.walk(); 
    auto g' <- this.y.walk();
    
    u:State?;         // previous state of m
    x:State?;         // current state of m
    y:Observation?;   // current observation of m
    u':State?;        // previous state of this
    x':State?;        // current state of this
    y':Observation?;  // current observation of this

    yield sum(proposeParameter(θ', θ));
    while true {
      w:Real <- 0.0;
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
        w <- w + sum(proposeTransition(x'!, u'!, θ', x!, u!, θ));
      } else {
        w <- w + sum(proposeInitial(x'!, θ', x!, θ));
      }
      u <- x;
      u' <- x';

      if g? {
        y <- g!;
      } else {
        error("previous state has incorrect number of checkpoints");
      }
      if g'? {  // is the next observation given?
        y' <- g'!;
      } else {
        o:Observation;
        this.y.pushBack(o);
        y' <- o;
      }
      w <- w + sum(proposeObservation(y'!, x'!, θ', y!, x!, θ));
      yield w;
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
    assert x.size() == y.size();
    w:Real <- 0.0;
    
    x:State;
    if this.x.empty() {
      w <- sum(initial(x, θ));
    } else {
      w <- sum(transition(x, this.x.back(), θ));
    }
    this.x.pushBack(x);
    
    y:Observation;
    w <- w + sum(observation(y, this.x.back(), θ));
    this.y.pushBack(y);
    
    return w;
  }

  function checkpoints() -> Integer? {
    /* one checkpoint for the parameters, then one for each time */
    return 1 + max(x.size(), y.size());
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
   * Observation model.
   *
   * - y: The observations, to be set.
   * - x: The current state.
   * - θ: The parameters.
   */
  fiber observation(y:Observation, x:State, θ:Parameter) -> Real {
    //
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

  /**
   * Observation proposal.
   *
   * - y: The observations, to be set.
   * - x: The current state.
   * - θ: The parameters.
   *
   * By default calls `observation(y, x, θ)`.
   */
  fiber proposeObservation(y:Observation, x:State, θ:Parameter) -> Real {
    observation(y, x, θ);
  }

  /**
   * Observation proposal.
   *
   * - y': The observations, to be set.
   * - x': The current state.
   * - θ': The parameters.
   * - y: The last observations.
   * - x: The last current state.
   * - θ: The last parameters.
   *
   * By default calls `proposeObservation(y', x', θ')`.
   */
  fiber proposeObservation(y':Observation, x':State, θ':Parameter,
      y:Observation, x:State, θ:Parameter) -> Real {
    proposeObservation(y', x', θ');
  }
}
