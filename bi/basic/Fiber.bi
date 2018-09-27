/**
 * Deep clone a fiber.
 */
function clone(f:Model!) -> Model! {
  cpp{{
  return f_.clone();
  }}
}
