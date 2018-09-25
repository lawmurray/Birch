class MultiObjectParameter {
  /**
   * Probability of death.
   */
  μ:Real;

  /**
   * Probability of detection.
   */
  ρ:Real;

  /**
   * Birth rate.
   */
  λ_0:Real;

  /**
   * Birch position lower bound.
   */
  l_0:Real[_];

  /**
   * Birth position upper bound.
   */
  u_0:Real[_];
  
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
   * Clutter position lower bound.
   */
  l_c:Real[_];

  /**
   * Clutter position upper bound.
   */
  u_c:Real[_];
  
  function read(reader:Reader) {
    μ <- reader.getReal("μ")!;
    ρ <- reader.getReal("ρ")!;
    λ_0 <- reader.getReal("λ_0")!;
    l_0 <- reader.getRealVector("l_0")!;
    u_0 <- reader.getRealVector("u_0")!;
    A <- reader.getRealMatrix("A")!;
    Q <- reader.getRealMatrix("Q")!;
    B <- reader.getRealMatrix("B")!;
    R <- reader.getRealMatrix("R")!;
    λ_c <- reader.getReal("λ_c")!;
    l_c <- reader.getRealVector("l_c")!;
    u_c <- reader.getRealVector("u_c")!;
  }
  
  function write(writer:Writer) {
    writer.setReal("μ", μ);
    writer.setReal("ρ", ρ);
    writer.setReal("λ_0", λ_0);
    writer.setRealVector("l_0", l_0);
    writer.setRealVector("u_0", u_0);
    writer.setRealMatrix("A", A);
    writer.setRealMatrix("Q", Q);
    writer.setRealMatrix("B", B);
    writer.setRealMatrix("R", R);
    writer.setReal("λ_c", λ_c);
    writer.setRealVector("l_c", l_c);
    writer.setRealVector("u_c", u_c);
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
    N <~ Poisson(θ.λ_0);
    for n:Integer in 1..N {
      o':Random<Real[_]>;
      o' ~ Uniform(θ.l_0, θ.u_0);
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
      o' ~ Uniform(θ.l_c, θ.u_c);
      y'.o.pushBack(o');
    }
  }
}
