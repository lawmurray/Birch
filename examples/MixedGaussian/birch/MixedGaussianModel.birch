/**
 * Parameter for MixedGaussianModel.
 */
class MixedGaussianParameter {
  /**
   * Linear-linear state transition matrix.
   */
  A:Real[_,_] <- [[1.0, 0.3, 0.0], [0.0, 0.92, -0.3], [0.0, 0.3, 0.92]];
  
  /**
   * Nonlinear-linear state transition matrix.
   */
  b:Real[_] <- [1.0, 0.0, 0.0];
  
  /**
   * Linear observation matrix.
   */
  c:Real[_] <- [1.0, -1.0, 1.0];
    
  /**
   * Linear state noise covariance.
   */
  Σ_x_l:Real[_,_] <- [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]];
  
  /**
   * Nonlinear state noise covariance.
   */
  σ2_x_n:Real <- 0.01;
  
  /**
   * Linear observation noise covariance.
   */
  σ2_y_l:Real <- 0.1;
  
  /**
   * Nonlinear observation noise covariance.
   */
  σ2_y_n:Real <- 0.1;

  function read(buffer:Buffer) {
    A <-? buffer.get("A", A);
    b <-? buffer.get("b", b);
    c <-? buffer.get("c", c);
    Σ_x_l <-? buffer.get("Σ_x_l", Σ_x_l);
    σ2_x_n <-? buffer.get("σ2_x_n", σ2_x_n);
    σ2_y_l <-? buffer.get("σ2_y_l", σ2_y_l);
    σ2_y_n <-? buffer.get("σ2_y_n", σ2_y_n);
  }

  function write(buffer:Buffer) {
    buffer.set("A", A);
    buffer.set("b", b);
    buffer.set("c", c);
    buffer.set("Σ_x_l", Σ_x_l);
    buffer.set("σ2_x_n", σ2_x_n);
    buffer.set("σ2_y_l", σ2_y_l);
    buffer.set("σ2_y_n", σ2_y_n);
  }
}

/**
 * State for MixedGaussianModel.
 */
class MixedGaussianState {
  /**
   * Nonlinear state.
   */
  n:Random<Real>;
  
  /**
   * Linear state.
   */
  l:Random<Real[_]>;

  function read(buffer:Buffer) {
    buffer.get("l", l);
    buffer.get("n", n);
  }

  function write(buffer:Buffer) {
    buffer.set("l", l);
    buffer.set("n", n);
  }
}

/**
 * Observation for MixedGaussianModel.
 */
class MixedGaussianObservation {
  /**
   * Nonlinear observation.
   */
  n:Random<Real>;
  
  /**
   * Linear observation.
   */
  l:Random<Real>;

  function read(buffer:Buffer) {
    buffer.get("l", l);
    buffer.get("n", n);
  }

  function write(buffer:Buffer) {
    buffer.set("l", l);
    buffer.set("n", n);
  }
}

/**
 * Linear-nonlinear state-space model. The delayed sampling feature of Birch
 * results in a Rao--Blackwellized particle filter with locally-optimal
 * proposal being applied to this model.
 *
 * The model is detailed in [Lindsten and Schön (2010)](../#references).
 */
class MixedGaussianModel < StateSpaceModel<MixedGaussianParameter,
    MixedGaussianState,MixedGaussianObservation> {
  fiber initial(x:MixedGaussianState, θ:MixedGaussianParameter) -> Event {
    x.n ~ Gaussian(0.0, 1.0);
    x.l ~ Gaussian(vector(0.0, 3), identity(3));
  }

  fiber transition(x':MixedGaussianState, x:MixedGaussianState,
      θ:MixedGaussianParameter) -> Event {
    x'.n ~ Gaussian(atan(x.n) + dot(θ.b, x.l), θ.σ2_x_n);
    x'.l ~ Gaussian(θ.A*x.l, θ.Σ_x_l);
  }
    
  fiber observation(y:MixedGaussianObservation, x:MixedGaussianState,
      θ:MixedGaussianParameter) -> Event {
    y.n ~ Gaussian(0.1*copysign(pow(x.n, 2.0), x.n), θ.σ2_y_n);
    y.l ~ Gaussian(dot(θ.c, x.l), θ.σ2_y_l);
  }    
}
