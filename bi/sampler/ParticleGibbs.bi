/**
 * Particle Gibbs sampler.
 */
class ParticleGibbs < ParticleFilter {
  /**
   * Traces of all paths in current iteration.
   */
  h:TraceHandler<DelayHandler>[_];

  /**
   * Trace of reference path from previous iteration.
   */
  x':TraceHandler<DelayHandler>;
   
  function initialize() {
    super.initialize();
    X1:List<Event>[N];
    X <- X1;
  }
  
  function start() {
    f' <- x'.walk();
    w[N] <- w[N] + handleReferenceStart(x[N].start(), f');
    parallel for auto n in 1..N-1 {
      w[n] <- w[n] + handleReferenceStart(x[n].start(), x'.walk());
    }
  }
  
  function step() {
    auto x0 <- x;
    x[N] <- clone<ForwardModel>(x0[a[N]]);
    w[N] <- w[N] + handleReferenceStep(x[N].step(), f');
    parallel for auto n in 1..N-1 {
      x[n] <- clone<ForwardModel>(x0[a[n]]);
      w[n] <- w[n] + handleStep(x[n].step());
    }
  }

  /**
   * Handle events when stepping the reference particle.
   */
  function handleReferenceStart(f:Event!, f':Event!) -> Real {
    auto w <- 0.0;
    while f? && f'? {
      auto evt <- f!;
      auto evt' <- f'!;
      if evt.isFactor() {
        w <- w + evt.observe();
      } else if evt.isRandom() {
        if evt.hasValue() {
          w <- w + evt.observe();
        } else {
          evt.propose(evt');  // okay to ignore proposal weight
        }
      }
    }
    return w;
  } 

  /**
   * Handle events when stepping the reference particle.
   */
  function handleReferenceStep(f:Event!, f':Event!) -> Real {
    auto w <- 0.0;
    while f? && f'? {
      auto evt <- f!;
      auto evt' <- f'!;
      if evt.isFactor() {
        w <- w + evt.observe();
      } else if evt.isRandom() {
        if evt.hasValue() {
          w <- w + evt.observe();
        } else {
          evt.propose(evt');  // okay to ignore proposal weight
        }
      }
    }
    return w;
  } 

  /**
   * Handle events when stepping the reference particle.
   */
  function handleReferenceStart(f:Event!, f':Event!) -> Real {
    auto w <- 0.0;
    while f? && f'? {
      auto evt <- f!;
      auto evt' <- f'!;
      if evt.isFactor() {
        w <- w + evt.observe();
      } else if evt.isRandom() {
        if evt.hasValue() {
          w <- w + evt.observe();
        } else {
          evt.propose(evt');  // okay to ignore proposal weight
        }
      }
    }
    return w;
  } 

  /**
   * Performs a Gibbs move on the parameters of the reference path.
   */
  function propose() {
    auto x <- clone<ForwardModel>(archetype);
    auto f <- x.start();
    auto f' <- x'.walk();
    handleSkipStart(f, f');
    for auto t in 1..T {
      handleReferenceStep(f, f');
    }
    x' <- X[1];
  }

  function resample() {
    super.resample();
    if !x0.empty() {
      a[N] <- N;
    }
  }

  function finish() {
    super.finish();
    x' <- X[b];
  }
}
