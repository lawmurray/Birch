/**
 * Pair of stacks used for the implementation of other containers, such as
 * Queue and Iterator.
 */
class DoubleStack<Type> {
  forward:StackNode<Type>?;
  backward:StackNode<Type>?;
  count:Integer <- 0;

  /**
   * Number of elements.
   */
  function size() -> Integer {
    return count;
  }

  /**
   * Is this empty?
   */
  function empty() -> Boolean {
    return count == 0;
  }

  /**
   * Clear all elements.
   */
  function clear() {
    forward <- nil;
    backward <- nil;
    count <- 0;
  }

  /**
   * Get the top element on the forward stack.
   *
   * - x: Value.
   */
  function topForward() -> Type {
    assert forward?;
    return forward!.x;
  }

  /**
   * Get the top element on the backward stack.
   *
   * - x: Value.
   */
  function topBackward() -> Type {
    assert backward?;
    return backward!.x;
  }

  /**
   * Push a new element on the forward stack.
   *
   * - x: Value.
   */
  function pushForward(x:Type) {
    node:StackNode<Type>(x);
    //node.next <- forward;
    if forward? {
      cpp{{
      node->next = std::move(self->forward.get());
      }}
    }
    forward <- node;
    count <- count + 1;
  }

  /**
   * Push a new element on the backward stack.
   *
   * - x: Value.
   */
  function pushBackward(x:Type) {
    node:StackNode<Type>(x);
    //node.next <- backward;
    if backward? {
      cpp{{
      node->next = std::move(self->backward.get());
      }}
    }
    backward <- node;
    count <- count + 1;
  }

  /**
   * Pop the top element from the forward stack and return it.
   */
  function popForward() -> Type {
    assert !empty();
    count <- count - 1;
    cpp{{
    auto x = std::move(self->forward.get()->x);
    self->forward = std::move(self->forward.get()->next);
    return x;
    }}
  }

  /**
   * Pop the top element from the backward stack and return it.
   */
  function popBackward() -> Type {
    assert !empty();
    count <- count - 1;
    cpp{{
    auto x = std::move(self->backward.get()->x);
    self->backward = std::move(self->backward.get()->next);
    return x;
    }}
  }

  /**
   * Move one element from the backward list to the forward list.
   */
  function oneForward() {
    pushForward(popBackward());
  }
  
  /**
   * Move one element from the forward list to the backward list.
   */
  function oneBackward() {
    pushBackward(popForward());
  }
  
  /**
   * Move all elements from the backward list to the forward list.
   */
  function allForward() {
    while backward? {
      oneForward();
    }
  }
  
  /**
   * Move all elements from the forward list to the backward list.
   */
  function allBackward() {
    while forward? {
      oneBackward();
    }
  }
}
