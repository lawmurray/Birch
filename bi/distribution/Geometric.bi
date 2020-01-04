/**
 * Create a Geometric distribution
 */
function Geometric(ρ:Expression<Real>) -> NegativeBinomial {
  m:NegativeBinomial(Boxed(1), ρ);
  return m;
}

/**
 * Create a Geometric distribution
 */
function Geometric(ρ:Real) -> NegativeBinomial {
  return Geometric(Boxed(ρ));
}
