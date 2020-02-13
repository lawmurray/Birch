/**
 * @file
 *
 * Defines the operations freeze, thaw, finish, etc. These are overloaded
 * with the correct behaviour for types internal to the Birch type system,
 * with appropriate defaults or null operations for external types that may
 * be encountered in C++ interactions.
 */
#pragma once

namespace libbirch {
class Label;

template<class T>
void freeze(T& o) {
  //
}

template<class T>
void thaw(T& o, Label* label) {
  //
}

template<class T>
void finish(T& o) {
  //
}

template<class T>
void freeze(std::function<T>& o) {
  assert(false);
  /// @todo Need to freeze any objects in the closure here, which may require
  /// a custom implementation of lambda functions in a similar way to fibers,
  /// rather than using std::function
}

template<class T>
void thaw(Label* label, std::function<T>& o) {
  assert(false);
  /// @todo Need to thaw any objects in the closure here, which may require
  /// a custom implementation of lambda functions in a similar way to fibers,
  /// rather than using std::function
}

template<class T>
void finish(std::function<T>& o) {
  assert(false);
  /// @todo Need to finish any objects in the closure here, which may require
  /// a custom implementation of lambda functions in a similar way to fibers,
  /// rather than using std::function
}

}
