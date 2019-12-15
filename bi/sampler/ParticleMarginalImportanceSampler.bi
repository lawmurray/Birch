/**
 * Particle marginal importance sampler.
 * 
 * The ParticleSampler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Sampler.svg"></object>
 * </center>
 */
class ParticleMarginalImportanceSampler < ParticleSampler {  
  fiber sample(model:Model) -> (Model, Real, Real[_], Real[_], Integer[_]) {
    x:Model[_];
    w:Real[_];
    lnormalize:Real;
    ess:Real;
    npropagations:Integer;

    lnormalizeAll:Vector<Real>;
    essAll:Vector<Real>;
    npropagationsAll:Vector<Integer>;

    for n in 1..nsamples {
      lnormalizeAll.clear();
      essAll.clear();
      npropagationsAll.clear();
    
      auto f <- filter.filter(model);
      while f? {
        (x, w, lnormalize, ess, npropagations) <- f!;
        
        lnormalizeAll.pushBack(lnormalize);
        essAll.pushBack(ess);
        npropagationsAll.pushBack(npropagations);
      }
      auto b <- ancestor(w);
      yield (x[b], lnormalizeAll.back(), lnormalizeAll.toArray(),
          essAll.toArray(), npropagationsAll.toArray());
    }
  }
}
