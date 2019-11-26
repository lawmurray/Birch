/**
 * Pair of stacks used for the implementation of other containers, such as
 * Queue and Iterator.
 */
class DoubleStack<Type> {
  forward:StackNode<Type>?;
  backward:StackNode<Type>?;
  forwardCount:Integer <- 0;
  backwardCount:Integer <- 0;

  /**
   * Number of elements.
   */
  function size() -> Integer {
    return forwardCount + backwardCount;
  }

  /**
   * Is this empty?
   */
  function empty() -> Boolean {
    return forwardCount + backwardCount == 0;
  }

  /**
   * Clear all elements.
   */
  function clear() {
    forward <- nil;
    backward <- nil;
    forwardCount <- 0;
    backwardCount <- 0;
  }

  /**
   * Get the top element on the forward stack.
   */
  function topForward() -> Type {
    assert forward?;
    return forward!.x;
  }

  /**
   * Get the top element on the backward stack.
   */
  function topBackward() -> Type {
    assert backward?;
    return backward!.x;
  }

  /**
   * Get and remove the whole forward stack.
   */
  function takeForward() -> StackNode<Type>? {
    forwardCount <- 0;
    cpp{{
    return std::move(self->forward);
    }}
  }

  /**
   * Get the top element on the backward stack.
   */
  function takeBackward() -> StackNode<Type>? {
    backwardCount <- 0;
    cpp{{
    return std::move(self->backward);
    }}
  }

  /**
   * Put the whole forward stack.
   */
  function putForward(forward:StackNode<Type>?, forwardCount:Integer) {
    this.forwardCount <- forwardCount;
    cpp{{
    self->forward.assign(context_, std::move(forward));
    }}
  }

  /**
   * Put the whole backward stack.
   */
  function putBackward(backward:StackNode<Type>?, backwardCount:Integer) {
    this.backwardCount <- backwardCount;
    cpp{{
    self->backward.assign(context_, std::move(backward));
    }}
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
      node->next.assign(context_, std::move(self->forward));
      }}
    }
    forward <- node;
    forwardCount <- forwardCount + 1;
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
      node->next.assign(context_, std::move(self->backward));
      }}
    }
    backward <- node;
    backwardCount <- backwardCount + 1;
  }

  /**
   * Pop the top element from the forward stack and return it.
   */
  function popForward() -> Type {
    assert !empty();
    if forwardCount == 0 {
      allForward();
    }
    forwardCount <- forwardCount - 1;
    cpp{{
    auto x = std::move(self->forward.get()->x);
    self->forward.assign(context_, std::move(self->forward.get()->next));
    return x;
    }}
  }

  /**
   * Pop the top element from the backward stack and return it.
   */
  function popBackward() -> Type {
    assert !empty();
    if backwardCount == 0 {
      allBackward();
    }
    backwardCount <- backwardCount - 1;
    cpp{{
    auto x = std::move(self->backward.get()->x);
    self->backward.assign(context_, std::move(self->backward.get()->next));
    return x;
    }}
  }

  /**
   * Move one element from the backward list to the forward list.
   */
  function oneForward() {
    cpp{{
    auto node = std::move(self->backward.get());
    self->backward.assign(context_, std::move(node->next));
    node->next.assign(context_, std::move(self->forward));
    self->forward.assign(context_, std::move(node));
    }}
    forwardCount <- forwardCount + 1;
    backwardCount <- backwardCount - 1;
  }
  
  /**
   * Move one element from the forward list to the backward list.
   */
  function oneBackward() {
    cpp{{
    auto node = std::move(self->forward.get());
    self->forward.assign(context_, std::move(node->next));
    node->next.assign(context_, std::move(self->backward));
    self->backward.assign(context_, std::move(node));
    }}
    forwardCount <- forwardCount - 1;
    backwardCount <- backwardCount + 1;
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
