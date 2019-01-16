/**
 * Particle marginal Metropolis--Hastings sampler.
 */
class ParticleMarginalMetropolisHastings < Sampler {
  /**
   * Inner sampler.
   */
  sampler:Sampler?;
  
  /**
   * Current state.
   */
  x:Model?;
  
  /**
   * Evaluation of target at current state, $\pi(x)$.
   */
  π:Real?;
    
  function sample(m:Model) -> (Model, Real) {
    /* if a number of checkpoints hasn't been explicitly provided, assume
     * Metropolis--Hastings for just the first, and the inner sampler for
     * the rest */
    if (!ncheckpoints?) {
      ncheckpoints <- 1;
    }

    /* Markov chain states and evaluations */
    x':Model <- m;
    π':Real;  // π(x')
    q:Real;   // q(x'|x)
    q':Real;  // q(x|x')
    
    if !x? {
      /* initial state */
      (x, π) <- sampler!.sample(x!);
    } else {
      /* subsequent states */
      x' <- clone<Model>(m);
      //x'.propose(x!);
      q' <- sum(x'.propose(x!));
      q <- sum(x!.propose(x'));
      
      (x', π') <- sampler!.sample(x');
      if simulate_bernoulli(exp(π' - π! + q - q')) {
        (x, π) <- (x', π');  // accept
      }
    }
    return (x!, 0.0);
  }

  function read(buffer:Buffer) {
    super.read(buffer);

    /* create the inner sampler */
    className:String?;
    samplerBuffer:Buffer? <- buffer.getObject("sampler");
    if (samplerBuffer?) {
      className <- samplerBuffer!.getString("class");
    }
    if (!className?) {
      className <- "ParticleFilter";
    }
    sampler <- Sampler?(make(className!));
    if (!sampler?) {
      error(className! + " is not a subtype of Sampler.");
    }
    buffer.get("sampler", sampler!);
  }
  
  function write(buffer:Buffer) {
    super.write(buffer);
    buffer.set("sampler", sampler!);
  }
}
