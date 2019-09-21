/**
 * Parameter for LinearGaussianModel.
 */
class LinearGaussianParameter {
  a:Real <- 0.8;
  b:Real <- 10.0;
  σ2_x:Real <- 1.0;
  σ2_y:Real <- 0.01;

  function read(buffer:Buffer) {
    a <-? buffer.get("a", a);
    b <-? buffer.get("b", b);
    σ2_x <-? buffer.get("σ2_x", σ2_x);
    σ2_y <-? buffer.get("σ2_y", σ2_y);
  }
  
  function write(buffer:Buffer) {
    buffer.set("a", a);
    buffer.set("b", b);
    buffer.set("σ2_x", σ2_x);
    buffer.set("σ2_y", σ2_y);
  }
}

/**
 * Linear-Gaussian state-space model. The delayed sampling feature of Birch
 * results in a Kalman filter being applied to this model.
 */
class LinearGaussianModel < StateSpaceModel<LinearGaussianParameter,
    Random<Real>,Random<Real>> {
  fiber initial(x:Random<Real>, θ:LinearGaussianParameter) -> Event {
    x ~ Gaussian(0.0, θ.σ2_x);
  }

  fiber transition(x':Random<Real>, x:Random<Real>,
      θ:LinearGaussianParameter) -> Event {
    x' ~ Gaussian(θ.a*x, θ.σ2_x);
  }

  fiber observation(y:Random<Real>, x:Random<Real>,
      θ:LinearGaussianParameter) -> Event {
    y ~ Gaussian(θ.b*x, θ.σ2_y);
  }
}
