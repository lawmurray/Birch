class ParticleMarginalMetropolisHastings < ParticleFilter {

  m:ForwardModel?; // previous model
  py:Real; // evidence of the previous model
  q:Real; // proposal weight of the previous model
  m':ForwardModel; // proposed model
  py':Real; // evidence of the proposed model
  q':Real; // proposal weight of the proposed model
  A:Real <- 0; // accepted samples
  R:Real <- 0; // rejected samples


  function initialize() {
    if !m? {
      // No pervious model, run as normal
      super.initialize();
    } else {
      /* There is a previous model, create particles from  proposal model */
      py <- sum(Z.walk()); // Save previous evidence
      
      m' <- clone<ForwardModel>(archetype!); // Create a proposal model from the archetype
      (q', q) <- m'.propose(m!); // Set the parameters of the proposed model from the previous model

      // Reset the particle filter
      Z.clear();
      ess.clear();
      memory.clear();
      elapsed.clear();

      w <- vector(0.0, N);
      a <- iota(1, N);
      o <- vector(1, N);

      // Clone the proposed model to all particles
      x1:Vector<ForwardModel>;
      x1.enlarge(N, clone<ForwardModel>(m'));
      x <- x1.toArray();

      parallel for auto n in 1..N {
        x[n] <- clone<ForwardModel>(x[n]);
      }

      tic();
    }
  }
  
  function finalize() {
    // The filter has drawn x[b]
    if m? {
      m' <- x[b];
      py' <- sum(Z.walk());
      if (log(simulate_uniform(0.0, 1.0)) < py' + q - py - q') {
        // accept and save for next iteration
        m <- clone<ForwardModel>(m');
        py <- py';
        A <- A + 1;
      } else {
        // reject and replace drawn model with previous model
        R <- R + 1;
        x[b] <- clone<ForwardModel>(m!);
      }
    } else {
      m <- clone<ForwardModel>(x[b]);
    }
    if verbose {
      stderr.print("acceptance rate: " + A/(A+R) + "\n");
    }
  }

}