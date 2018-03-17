/**
 * Linear-nonlinear state-space model parameter.
 */
class LinearNonlinearSSMParameter < Model {
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
}

/**
 * Linear-nonlinear state-space model state.
 */
class LinearNonlinearSSMState < State {
  /**
   * Nonlinear state.
   */
  x_n:Random<Real[_]>;
  
  /**
   * Linear state.
   */
  x_l:Random<Real[_]>;

  /**
   * Nonlinear observation.
   */
  y_n:Random<Real[_]>;
  
  /**
   * Linear observation.
   */
  y_l:Random<Real[_]>;

  fiber simulate(θ:LinearNonlinearSSMParameter) -> Real! {
    x_n ~ Gaussian(vector(0.0, 1), identity(1));
    x_l ~ Gaussian(vector(0.0, 3), identity(3));
    y_n ~ Gaussian([0.1*copysign(pow(scalar(x_n), 2.0), scalar(x_n))], θ.Σ_y_n);
    y_l ~ Gaussian(θ.C*x_l, θ.Σ_y_l);
  }

  fiber simulate(χ:LinearNonlinearSSMState, θ:LinearNonlinearSSMParameter) -> Real! {    
    x_n ~ Gaussian([atan(scalar(χ.x_n))] + θ.B*χ.x_l, θ.Σ_x_n);
    x_l ~ Gaussian(θ.A*χ.x_l, θ.Σ_x_l);
    y_n ~ Gaussian(vector(0.1*copysign(pow(scalar(x_n), 2.0), scalar(x_n)), 1), θ.Σ_y_n);
    y_l ~ Gaussian(θ.C*x_l, θ.Σ_y_l);
  }

  function input(reader:Reader) {
    y_l <- reader.getRealVector("y_l");
    y_n <- reader.getRealVector("y_n");
  }

  function output(writer:Writer) {
    writer.setObject();
    writer.setRealVector("x_l", x_l);
    writer.setRealVector("x_n", x_n);
  }
}

/**
 * Linear-nonlinear state-space model. When applied to this model, SMC with
 * delayed sampling yields a Rao--Blackwellized particle filter with
 * locally-optimal proposal.
 */
class LinearNonlinearSSM = MarkovModel<LinearNonlinearSSMState,
    LinearNonlinearSSMParameter>;
