/**
 * Parameter for LinearGaussianModel.
 */
class LinearGaussianParameter {
  a:Real <- 0.8;
  σ2_x:Real <- 1.0;
  σ2_y:Real <- 0.1;
}

/**
 * Linear-Gaussian state-space model. The delayed sampling feature of Birch
 * results in a Kalman filter being applied to this model.
 */
class LinearGaussianModel < StateSpaceModel<LinearGaussianParameter,Random<Real>,Random<Real>> {
  fiber initial(x':Random<Real>, θ:LinearGaussianParameter) -> Real {
    x' ~ Gaussian(0.0, θ.σ2_x);
  }

  fiber transition(x':Random<Real>, x:Random<Real>, θ:LinearGaussianParameter) -> Real {
    x' ~ Gaussian(θ.a*x, θ.σ2_x);
  }

  fiber observation(y':Random<Real>, x:Random<Real>, θ:LinearGaussianParameter) -> Real {
    y' ~ Gaussian(x, θ.σ2_y);
  }
}
