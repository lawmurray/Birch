/**
 * Deep clone an object or fiber.
 */
function clone<Type>(o:Type) -> Type {
  cpp{{
  return pop_context(push_context(o_).clone());
  }}
}
