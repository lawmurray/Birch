/**
 * (Doubly) linked list.
 */
class LinkedList {
  head:LinkedListNode?;
  tail:LinkedListNode?;
  count:Integer <- 0;

  function isEmpty() -> Boolean {
    return !head?;
  }

  function empty() {
    obj:Object? <- pop();
    while (obj?) {
      obj <- pop();
    }
  }

  function push(obj:Object) {
    node:LinkedListNode <- LinkedListNode(obj);
    if (tail?) {
      tail!.next <- node;
      node.prev <- tail;
    } else {
      head <- node;
    }
    tail <- node;
    count <- count + 1;
  }

  function unshift(obj:Object) {
    node:LinkedListNode <- LinkedListNode(obj);
    if (head?) {
      head!.prev <- node;
      node.next <- head;
    } else {
      tail <- node;
    }
    head <- node;
    count <- count + 1;
  }

  function pop() -> Object? {
    obj:Object?;
    curr:LinkedListNode? <- tail;
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

  function pop(i:Integer) -> Object? {
    obj:Object?;
    if (i > count) {
      return obj;
    }
    curr:LinkedListNode?;
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

  function shift() -> Object? {
    return pop(1);
  }

  fiber walk() -> Object! {
    curr:LinkedListNode? <- head;
    while (curr?) {
      yield curr!.obj;
      curr <- curr!.next;
    }
  }
}
