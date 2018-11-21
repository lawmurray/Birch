class Multi < StateSpaceModel<Global,List<Track>,List<Random<Real[_]>>> {
  /**
   * Current time.
   */
  t:Integer <- 0;

  /**
   * All tracks up to current time.
   */
  z:List<Track>;

  fiber transition(x':List<Track>, x:List<Track>, θ:Global) -> Real {
    /* update time */
    t <- t + 1;

    /* move current objects */
    auto track <- x.walk();
    while track? {
      ρ:Real <- pmf_poisson(t - track!.t - 1, θ.τ);
      R:Real <- 1.0 - cdf_poisson(t - track!.t - 1, θ.τ) + ρ;
      s:Boolean;
      s <~ Bernoulli(1.0 - ρ/R);  // does the object survive?
      if s {
        track!.step();
        x'.pushBack(track!);
      }
    }
    
    /* birth new objects */
    N:Integer;
    N <~ Poisson(θ.λ);
    for n:Integer in 1..N {
      track:Track;
      track.t <- t;
      track.θ <- θ;
      track.start();
      x'.pushBack(track);
      z.pushBack(track);
    }
  }

  fiber observation(y:List<Random<Real[_]>>, x:List<Track>, θ:Global) -> Real {
    if !y.empty() {
      association(y, x, θ);
    } else {
      /* clutter */
      N:Integer;
      N <~ Poisson(θ.μ);
      for n:Integer in 1..(N + 1) {
        clutter:Random<Real[_]>;
        clutter <~ Uniform(θ.l, θ.u);
        y.pushBack(clutter);
      }
    }
  }

  fiber association(y:List<Random<Real[_]>>, x:List<Track>, θ:Global) -> Real {
    K:Integer <- 0;  // number of detections
    auto track <- x.walk();
    while track? {
      if track!.y.back().hasDistribution() {
        /* object is detected, compute proposal */
        K <- K + 1;
        q:Real[y.size()];
        n:Integer <- 1;
        auto detection <- y.walk();
        while detection? {
          q[n] <- track!.y.back().pdf(detection!);
          n <- n + 1;
        }
        Q:Real <- sum(q);
          
        /* propose an association */
        if Q > 0.0 {
          q <- q/Q;
          n <~ Categorical(q);  // choose an observation
          yield track!.y.back().realize(y.get(n));  // likelihood
          yield -log(q[n]);  // proposal correction
          y.erase(n);  // remove the observation for future associations
        } else {
          yield -inf;  // detected, but all likelihoods (numerically) zero
        }
      }

      /* factor in prior probability of hypothesis */
      yield -lrising(y.size() + 1, K);  // prior correction
    }
    
    /* clutter */
    y.size() - 1 ~> Poisson(θ.μ);
    auto clutter <- y.walk();
    while clutter? {
      clutter! ~> Uniform(θ.l, θ.u);
    }
  }
  
  function read(reader:Reader) {
    θ.read(reader.getObject("θ"));
    y1.read(reader.getObject("y"));
  }
  
  function write(writer:Writer) {
    θ.write(writer.setObject("θ"));
    z.write(writer.setObject("z"));
    y.write(writer.setObject("y"));
  }
}
