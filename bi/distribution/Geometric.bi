/**
 * Create a Geometric distribution
 */
final class Geometric(ρ:Expression<Real>) < NegativeBinomial(Boxed(1), ρ) {}

/**
 * Create a Geometric distribution
 */
function Geometric(ρ:Expression<Real>) -> Geometric {
  m:Geometric(ρ);
  return m;
}

/**
 * Create a Geometric distribution
 */
function Geometric(ρ:Real) -> Geometric {
  return Geometric(Boxed(ρ));
}