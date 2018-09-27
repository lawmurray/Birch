class Track = StateSpaceVariate<Global,Random<Real[_]>,Random<Real[_]>>;
class Detection = Random<Real[_]>;

class TrackModel < StateSpaceModel<Track> {
  fiber initial(x':Random<Real[_]>, θ:Global) -> Real {
    auto μ <- vector(0.0, 3*length(θ.l));
    μ[1..2] <~ Uniform(θ.l, θ.u);
    x' ~ Gaussian(μ, θ.M);
  }
  
  fiber transition(x':Random<Real[_]>, x:Random<Real[_]>, θ:Global) -> Real {
    x' ~ Gaussian(θ.A*x, θ.Q);
  }

  fiber observation(y':Random<Real[_]>, x:Random<Real[_]>, θ:Global) -> Real {
    y' ~ Gaussian(θ.B*x, θ.R);
  }
}

class Multi = StateSpaceVariate<Global,List<Track>,List<Detection>>;

class MultiModel < StateSpaceModel<Multi> {
  m:TrackModel;

  fiber transition(x':List<Track>, x:List<Track>, θ:Global) -> Real {
    /* move current objects */
    auto track <- x.walk();
    while track? {
      s:Boolean;
      s <~ Bernoulli(θ.s);
      if (s) {
        m.step(track!);
        x'.pushBack(track!);
      }
    }
    
    /* birth new objects */
    N:Integer;
    N <~ Poisson(θ.λ);
    for n:Integer in 1..N {
      track:Track;
      track.θ <- θ;
      m.start(track);
      x'.pushBack(track);
    }
  }

  fiber observation(y':List<Detection>, x:List<Track>, θ:Global) -> Real {
    /* an empty list of observations is interpreted as missing observations,
     * and so they will be generated, otherwise data association to tracks is
     * used */
    associate:Boolean <- !y'.empty();
    n:Integer <- 0;
  
    /* current objects */
    auto track <- x.walk();
    while track? {
      d:Boolean;
      d <~ Bernoulli(θ.d);
      if d {
        /* object is detected */
        if associate {
          /* associate this object with an observation */
          p:Real[y'.size()];
          auto detection <- y'.walk();
          n <- 1;
          while detection? {
            p[n] <- track!.y.back().pdf(detection!);
            n <- n + 1;
          }
          P:Real <- sum(p);          
          if P > 0.0 {  // check
            p <- p/P;
            n <~ Categorical(p);
            yield track!.y.back().pdf(y'.get(n)) - log(p[n]);
            y'.erase(n);
          }
        } else {
          /* simulate an observation for this object */
          y'.pushBack(track!.y.back());
        }
      }
    }

    /* clutter */
    if associate {
      y'.size() ~> Poisson(θ.μ);
      auto detection <- y'.walk();
      while detection? {
        detection! ~> Uniform(θ.l, θ.u);
      }
    } else {
      N:Integer;
      N <~ Poisson(θ.μ);
      for n:Integer in 1..N {
        detection:Detection;
        detection <~ Uniform(θ.l, θ.u);
        y'.pushBack(detection);
      }
    }
  }
}
