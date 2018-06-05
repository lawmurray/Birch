/**
 * Parameter model for mixed linear-nonlinear Gaussian state-space model.
 */
class MixedGaussianParameter < Parameter {
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
