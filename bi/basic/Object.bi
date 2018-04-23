/**
 * Root class of all other classes.
 */
class Object;

/**
 * Identity comparison.
 */
operator (x:Object == y:Object) -> Boolean;

/**
 * Identity comparison.
 */
operator (x:Object != y:Object) -> Boolean;

/**
 * Identity conversion.
 */
function Object(o:Object) -> Object {
  return o;
}
