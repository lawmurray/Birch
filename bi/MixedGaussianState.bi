/**
 * State for MixedGaussianState.
 */
class MixedGaussianState < State {
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

  fiber initial(θ:MixedGaussianParameter) -> Real {
    x_n ~ Gaussian(vector(0.0, 1), identity(1));
    x_l ~ Gaussian(vector(0.0, 3), identity(3));
    y_n ~ Gaussian([0.1*copysign(pow(scalar(x_n), 2.0), scalar(x_n))], θ.Σ_y_n);
    y_l ~ Gaussian(θ.C*x_l, θ.Σ_y_l);
  }

  fiber transition(χ:MixedGaussianState, θ:MixedGaussianParameter) -> Real {    
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
