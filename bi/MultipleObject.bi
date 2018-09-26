class MultiObjectParameter {
  /**
   * Lower corner of domain of interest.
   */
  l:Real[_];

  /**
   * Upper corner of domain of interest.
   */
  u:Real[_];

  /**
   * Probability of survival.
   */
  s:Real;

  /**
   * Probability of detection.
   */
  d:Real;

  /**
   * Initial value covariance.
   */
  M:Real[_,_];
  
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
   * Birth rate.
   */
  λ:Real;

  /**
   * Clutter rate.
   */
  μ:Real;
  
  function read(reader:Reader) {
    l <- reader.get("l", l)!;
    u <- reader.get("u", u)!;
    s <- reader.get("s", s)!;
    d <- reader.get("d", d)!;
    M <- reader.get("M", M)!;
    A <- reader.get("A", A)!;
    Q <- reader.get("Q", Q)!;
    B <- reader.get("B", B)!;
    R <- reader.get("R", R)!;
    λ <- reader.get("λ", λ)!;
    μ <- reader.get("μ", μ)!;
  }
  
  function write(writer:Writer) {
    writer.set("l", l);
    writer.set("u", u);
    writer.set("s", s);
    writer.set("d", d);
    writer.set("M", M);
    writer.set("A", A);
    writer.set("Q", Q);
    writer.set("B", B);
    writer.set("R", R);
    writer.set("λ", λ);
    writer.set("μ", μ);
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
  o:List<Random<Real[_]>>?;

  function read(reader:Reader) {
    o:List<Random<Real[_]>>;
    o.read(reader);
    this.o <- o;
  }

  function write(writer:Writer) {
    o!.write(writer);
  }
}

class MultiObjectVariate = StateSpaceVariate<MultiObjectParameter,MultiObjectState,MultiObjectObservation>;

/**
 * Model for Multi objects.
 */
class MultiObjectModel < StateSpaceModel<MultiObjectVariate> {
  fiber f(x':MultiObjectState, x:MultiObjectState, θ:MultiObjectParameter) -> Real {
    /* move current objects */
    auto o <- x.o.walk();
    while o? {
      s:Boolean;
      s <~ Bernoulli(θ.s);
      if (s) {
        o':Random<Real[_]>;
        o' ~ Gaussian(θ.A*o!, θ.Q);
        x'.o.pushBack(o');
      }
    }
    
    /* birth new objects */
    N:Integer;
    N <~ Poisson(θ.λ);
    for n:Integer in 1..N {
      auto μ <- vector(0.0, 3*length(θ.l));
      μ[1..2] <~ Uniform(θ.l, θ.u);
      o':Random<Real[_]>;
      o' ~ Gaussian(μ, θ.M);
      x'.o.pushBack(o');
    }
  }

  fiber g(y':MultiObjectObservation, x:MultiObjectState, θ:MultiObjectParameter) -> Real {
    if y'.o? {
      /* current objects */
      auto r <- y'.o!.copy();
      auto o <- x.o.walk();
      while o? {
        d:Boolean;
        d <~ Bernoulli(θ.d);  // is this object detected?
        if (d) {
          p:Real[r.size()];
          i:Integer <- 1;
          auto f <- r.walk();
          while f? {
            p[i] <- Gaussian(θ.B*o!, θ.R).pdf(f!);
            i <- i + 1;
          }
          P:Real <- sum(p);
          if P > 0.0 {
            p <- p/P;
            i <~ Categorical(p);            // propose an association
            r.get(i) ~> Gaussian(θ.B*o!, θ.R);  // associate
            r.erase(i);                     // observation is now associated
            i ~> Categorical(p);            // proposal correction
          }
        }
      }
      
      /* unassociated observations are just clutter */
      r.size() ~> Poisson(θ.μ);
      auto f <- r.walk();
      while f? {
        f! ~> Uniform(θ.l, θ.u);
      }
    } else {
      /* initialise with empty list */
      l:List<Random<Real[_]>>;
      y'.o <- l;
    
      /* current objects */
      auto o <- x.o.walk();
      while o? {
        d:Boolean;
        d <~ Bernoulli(θ.d);  // is this object detected?
        if (d) {
          o':Random<Real[_]>;
          o' ~ Gaussian(θ.B*o!, θ.R);
          y'.o!.pushBack(o');
        }
      }

      /* clutter */
      N:Integer;
      N <~ Poisson(θ.μ);
      for n:Integer in 1..N {
        o':Random<Real[_]>;
        o' ~ Uniform(θ.l, θ.u);
        y'.o!.pushBack(o');
      }
    }
  }
}
