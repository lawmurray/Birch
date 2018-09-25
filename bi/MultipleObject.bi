class MultiObjectParameter {
  /**
   * Birth rate.
   */
  λ:Real;

  /**
   * Probability of death.
   */
  μ:Real;

  /**
   * Probability of detection.
   */
  ρ:Real;

  /**
   * Initial state mean.
   */
  μ_0:Real[_];

  /**
   * Initial state covariance.
   */
  Σ_0:Real[_,_];
  
  /**
   * Transition matrix.
   */
  A:Real[_,_];

  /**
   * State noise covariance matrix.
   */
  Q:Real[_,_];

  /**
   * Observation matrix.
   */
  B:Real[_,_];

  /**
   * Observation noise covariance matrix.
   */
  R:Real[_,_];

  /**
   * Clutter rate.
   */
  λ_c:Real;

  /**
   * Clutter mean.
   */
  μ_c:Real[_];

  /**
   * Clutter covariance.
   */
  Σ_c:Real[_,_];
  
  function read(reader:Reader) {
    λ <- reader.getReal("λ")!;
    μ <- reader.getReal("μ")!;
    ρ <- reader.getReal("ρ")!;
    μ_0 <- reader.getRealVector("μ_0")!;
    Σ_0 <- reader.getRealMatrix("Σ_0")!;
    A <- reader.getRealMatrix("A")!;
    Q <- reader.getRealMatrix("Q")!;
    B <- reader.getRealMatrix("B")!;
    R <- reader.getRealMatrix("R")!;
    λ_c <- reader.getReal("λ_c")!;
    μ_c <- reader.getRealVector("μ_c")!;
    Σ_c <- reader.getRealMatrix("Σ_c")!;
  }
  
  function write(writer:Writer) {
    writer.setReal("λ", λ);
    writer.setReal("μ", μ);
    writer.setReal("ρ", ρ);
    writer.setRealVector("μ_0", μ_0);
    writer.setRealMatrix("Σ_0", Σ_0);
    writer.setRealMatrix("A", A);
    writer.setRealMatrix("Q", Q);
    writer.setRealMatrix("B", B);
    writer.setRealMatrix("R", R);
    writer.setReal("λ_c", λ_c);
    writer.setRealVector("μ_c", μ_c);
    writer.setRealMatrix("Σ_c", Σ_c);
  }
}

class MultiObjectState {
  o:List<Random<Real[_]>>;
  
  function read(reader:Reader) {
    o.read(reader);
  }

  function write(writer:Writer) {
    o.write(writer);
  }
}

class MultiObjectObservation {
  o:List<Random<Real[_]>>;

  function read(reader:Reader) {
    o.read(reader);
  }

  function write(writer:Writer) {
    o.write(writer);
  }
}

class MultiObjectVariate = StateSpaceVariate<MultiObjectParameter,MultiObjectState,MultiObjectObservation>;

/**
 * Model for Multi objects.
 */
class MultiObjectModel < StateSpaceModel<MultiObjectVariate> {
  fiber m(x':MultiObjectState, θ:MultiObjectParameter) -> Real {
    
  }
  
  fiber f(x':MultiObjectState, x:MultiObjectState, θ:MultiObjectParameter) -> Real {
    /* move current objects */
    auto o <- x.o.walk();
    while o? {
      d:Boolean;
      d <~ Bernoulli(θ.μ);  // does this object die?
      if (!d) {
        o':Random<Real[_]>;
        o' ~ Gaussian(θ.A*o!, θ.Q);
        x'.o.pushBack(o');
      }
    }
    
    /* birth new objects */
    N:Integer;
    N <~ Poisson(θ.λ);
    for n:Integer in 1..N {
      o':Random<Real[_]>;
      o' ~ Gaussian(θ.μ_0, θ.Σ_0);
      x'.o.pushBack(o');
    }
  }

  fiber g(y':MultiObjectObservation, x:MultiObjectState, θ:MultiObjectParameter) -> Real {
    /* observe current objects */
    auto o <- x.o.walk();
    while o? {
      d:Boolean;
      d <~ Bernoulli(θ.ρ);  // is this object detected?
      if (d) {
        o':Random<Real[_]>;
        o' ~ Gaussian(θ.B*o!, θ.R);
        y'.o.pushBack(o');
      }
    }

    /* clutter */
    N:Integer;
    N <~ Poisson(θ.λ_c);
    for n:Integer in 1..N {
      o':Random<Real[_]>;
      o' ~ Gaussian(θ.μ_c, θ.Σ_c);
      y'.o.pushBack(o');
    }
  }
}
