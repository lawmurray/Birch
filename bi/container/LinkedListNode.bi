/**
 * Linked list node
 */
class LinkedListNode {
  obj:Object;
  next:LinkedListNode?;
  prev:LinkedListNode?;
}

/**
 * Create a linked list node.
 */
function LinkedListNode(obj:Object) -> LinkedListNode {
  node:LinkedListNode;
  node.obj <- obj;
  return node;
}
