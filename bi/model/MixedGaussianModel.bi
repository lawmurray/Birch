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
  B:Real[_,_] <- [[1.0, 0.0, 0.0]];
  
  /**
   * Linear observation matrix.
   */
  C:Real[_,_] <- [[1.0, -1.0, 1.0]];
    
  /**
   * Linear state noise covariance.
   */
  Σ_x_l:Real[_,_] <- [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]];
  
  /**
   * Nonlinear state noise covariance.
   */
  Σ_x_n:Real[_,_] <- [[0.01]];
  
  /**
   * Linear observation noise covariance.
   */
  Σ_y_l:Real[_,_] <- [[0.1]];
  
  /**
   * Nonlinear observation noise covariance.
   */
  Σ_y_n:Real[_,_] <- [[0.1]];

  function write(buffer:Buffer) {
    buffer.set("A", A);
    buffer.set("B", B);
    buffer.set("C", C);
    buffer.set("Σ_x_l", Σ_x_l);
    buffer.set("Σ_x_n", Σ_x_n);
    buffer.set("Σ_y_l", Σ_y_l);
    buffer.set("Σ_y_n", Σ_y_n);
  }
}

/**
 * State for MixedGaussianModel.
 */
class MixedGaussianState {
  /**
   * Nonlinear state.
   */
  n:Random<Real[_]>;
  
  /**
   * Linear state.
   */
  l:Random<Real[_]>;

  function read(buffer:Buffer) {
    l <- buffer.getRealVector("l");
    n <- buffer.getRealVector("n");
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
  n:Random<Real[_]>;
  
  /**
   * Linear observation.
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
 * Linear-nonlinear state-space model. The delayed sampling feature of Birch
 * results in a Rao--Blackwellized particle filter with locally-optimal
 * proposal being applied to this model.
 *
 * The model is detailed in [Lindsten and Schön (2010)](../#references).
 */
class MixedGaussianModel < StateSpaceModel<MixedGaussianParameter,
    MixedGaussianState,MixedGaussianObservation> {
  fiber initial(x':MixedGaussianState, θ:MixedGaussianParameter) -> Real {
    x'.n ~ Gaussian(vector(0.0, 1), identity(1));
    x'.l ~ Gaussian(vector(0.0, 3), identity(3));
  }

  fiber transition(x':MixedGaussianState, x:MixedGaussianState,
      θ:MixedGaussianParameter) -> Real {    
    x'.n ~ Gaussian([atan(scalar(x.n))] + θ.B*x.l, θ.Σ_x_n);
    x'.l ~ Gaussian(θ.A*x.l, θ.Σ_x_l);
  }
    
  fiber observation(y':MixedGaussianObservation, x:MixedGaussianState,
      θ:MixedGaussianParameter) -> Real {
    y'.n ~ Gaussian(vector(0.1*copysign(pow(scalar(x.n), 2.0), scalar(x.n)), 1), θ.Σ_y_n);
    y'.l ~ Gaussian(θ.C*x.l, θ.Σ_y_l);
  }    
}
