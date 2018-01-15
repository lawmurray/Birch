/**
 * (Doubly) linked list.
 */
class LinkedList<Type> {
  head:LinkedListNode<Type>?;
  tail:LinkedListNode<Type>?;
  count:Integer <- 0;

  function isEmpty() -> Boolean {
    return !head?;
  }

  function empty() {
    obj:Type? <- pop();
    while (obj?) {
      obj <- pop();
    }
  }

  function push(obj:Type) {
    node:LinkedListNode<Type>;
    node.obj <- obj;
    if (tail?) {
      tail!.next <- node;
      node.prev <- tail;
    } else {
      head <- node;
    }
    tail <- node;
    count <- count + 1;
  }

  function unshift(obj:Type) {
    node:LinkedListNode<Type>;
    node.obj <- obj;
    if (head?) {
      head!.prev <- node;
      node.next <- head;
    } else {
      tail <- node;
    }
    head <- node;
    count <- count + 1;
  }

  function pop() -> Type? {
    obj:Type?;
    curr:LinkedListNode<Type>? <- tail;
    if (curr?) {
      tail <- curr!.prev;
      if (tail?) {
        tail!.next <- nil;
      } else {
        head <- nil;
      }
      curr!.prev <- nil;
      obj <- curr!.obj;
      count <- count - 1;
    }
    return obj;
  }

  function pop(i:Integer) -> Type? {
    obj:Type?;
    if (i > count) {
      return obj;
    }
    curr:LinkedListNode<Type>?;
    if (2*i <= count) {
      curr <- head;
      for (j:Integer in 1..i-1) {
        if (curr?) {
          curr <- curr!.next;
        }
      }
    } else {
      curr <- tail;
      for (j:Integer in 1..count-i) {
        if (curr?) {
          curr <- curr!.prev;
        }
      }
    }
    if (curr?) {
      if (curr!.prev?) {
        curr!.prev!.next <- curr!.next;
      } else {
        head <- curr!.next;
      }
      if (curr!.next?) {
        curr!.next!.prev <- curr!.prev;
      } else {
        tail <- curr!.prev;
      }
      curr!.prev <- nil;
      curr!.next <- nil;
      count <- count - 1;
      obj <- curr!.obj;
    }
    return obj;
  }

  function shift() -> Type? {
    return pop(1);
  }

  fiber walk() -> Type! {
    curr:LinkedListNode<Type>? <- head;
    while (curr?) {
      yield curr!.obj;
      curr <- curr!.next;
    }
  }
}
