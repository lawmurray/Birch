/**
 * Create a Geometric distribution
 */
function Geometric(ρ:Expression<Real>) -> NegativeBinomial {
  return NegativeBinomial(Boxed(1), ρ);
}

/**
 * Create a Geometric distribution
 */
function Geometric(ρ:Real) -> NegativeBinomial {
  return Geometric(Boxed(ρ));
}
