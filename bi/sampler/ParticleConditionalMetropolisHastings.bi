class ParticleConditionalMetropolisHastings < ConditionalParticleFilter {

  m:ForwardModel?; // previous model
  py:Real; // evidence of the previous model
  q:Real; // proposal weight of the previous model
  m':ForwardModel; // proposed model
  py':Real; // evidence of the proposed model
  q':Real; // proposal weight of the proposed model
  A:Real <- 0; // accepted samples
  R:Real <- 0; // rejected samples


  function initialize() {
    if h'? {
      /* There is a previous model, save previous evidence and model*/
      py <- sum(Z.walk()); // Save previous evidence
      m <- clone<ForwardModel>(x[b]);
    }
      super.initialize();
  }

  function start() {
    if !h'? {
      super.start();
    } else {
      /* There is a previous model */
      auto x <- clone<ForwardModel>(archetype!); // create proposal model
      auto h <- h'!;
      h.rewind();
      h.setMode(PROPOSE_IMMEDIATE);
      (q', q) <- x.propose(m!); // Set the parameters of the proposed model from the previous model
      x.start();

      /* clone to all particles */
      auto forwardCount <- h.trace.forwardCount;
      auto forward <- h.trace.takeForward();
      h.setMode(PLAY_DELAY);
      for auto n in 1..N {
        this.x[n] <- clone<ForwardModel>(x);
      }
      h <- this.x[b].h;
      h.setMode(REPLAY_DELAY);
      h.trace.putForward(forward,forwardCount);
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
