/**
 * @file
 */
#pragma once

namespace libbirch {
class LazyLabel;

/*
 * Is this a value type?
 */
template<class T>
struct is_value {
  static const bool value = true;
};

/*
 * Is this a pointer type?
 */
template<class T>
struct is_pointer {
  static const bool value = false;
};

/**
 * Recursively freeze objects. This is used when an object is lazily cloned,
 * to ensure that the object, and all other objects reachable from it, are
 * no longer modifiable.
 */
template<class T>
void freeze(T& o) {
  //static_assert(is_value<T>::value, "unimplemented freeze()");
}

/**
 * Shallow thaw object. This is used when an object with only one remaining
 * reference is copied; instead of actually copying it is updated with a new
 * label for reuse.
 */
template<class T>
void thaw(T& o, LazyLabel* label) {
  //static_assert(is_value<T>::value, "unimplemented thaw()");
}

/**
 * Recursively finish objects. This is used when an object is lazily cloned,
 * to ensure that that object, and all other objects reachable from it, are
 * no longer modifiable.
 */
template<class T>
void finish(T& o) {
  //static_assert(is_value<T>::value, "unimplemented finish()");
}

}

#include "libbirch/Shared.hpp"
#include "libbirch/Weak.hpp"
#include "libbirch/Init.hpp"
#include "libbirch/Optional.hpp"
#include "libbirch/Fiber.hpp"
#include "libbirch/Array.hpp"
#include "libbirch/Tuple.hpp"

namespace libbirch {
template<class T>
struct is_value<Shared<T>> {
  static const bool value = false;
};

template<class T>
struct is_value<Weak<T>> {
  static const bool value = false;
};

template<class T>
struct is_value<Init<T>> {
  static const bool value = false;
};

template<class T>
struct is_value<Fiber<T>> {
  static const bool value = false;
};

template<class T, class F>
struct is_value<Array<T,F>> {
  static const bool value = is_value<T>::value;
};

template<class T>
struct is_value<std::initializer_list<T>> {
  static const bool value = is_value<T>::value;
};

template<class T>
struct is_value<Optional<T>> {
  static const bool value = is_value<T>::value;
};

template<class T>
struct is_value<std::function<T>> {
  static const bool value = false;
};

template<class Arg>
struct is_value<Tuple<Arg>> {
  static const bool value = is_value<Arg>::value;
};

template<class Arg, class ... Args>
struct is_value<Tuple<Arg,Args...>> {
  static const bool value = is_value<Arg>::value && is_value<Tuple<Args...>>::value;
};

template<class T>
struct is_pointer<Shared<T>> {
  static const bool value = true;
};

template<class T>
struct is_pointer<Weak<T>> {
  static const bool value = true;
};

template<class T>
struct is_pointer<Init<T>> {
  static const bool value = true;
};

template<class T>
void freeze(Shared<T>& o) {
  o.freeze();
}

template<class T>
void freeze(Weak<T>& o) {
  o.freeze();
}

template<class T>
void freeze(Init<T>& o) {
  o.freeze();
}

template<class T>
void freeze(Fiber<T>& o) {
  o.freeze();
}

template<class T, class F>
void freeze(Array<T,F>& o) {
  if (!is_value<T>::value) {
    auto iter = o.begin();
    auto last = iter + o.size();
    for (; iter != last; ++iter) {
      freeze(*iter);
    }
  }
}

template<class T>
void freeze(Optional<T>& o) {
  if (!is_value<T>::value && o.query()) {
    freeze(o.get());
  }
}

template<class T>
void freeze(std::function<T>& o) {
  assert(false);
  /// @todo Need to freeze any objects in the closure here, which may require
  /// a custom implementation of lambda functions in a similar way to fibers,
  /// rather than using std::function
}

template<class... Args>
void freeze(Tuple<Args...>& o) {
  o.freeze();
}

template<class T>
void thaw(Shared<T>& o, LazyLabel* label) {
  o.thaw(label);
}

template<class T>
void thaw(Weak<T>& o, LazyLabel* label) {
  o.thaw(label);
}

template<class T>
void thaw(Init<T>& o, LazyLabel* label) {
  o.thaw(label);
}

template<class T>
void thaw(Fiber<T>& o, LazyLabel* label) {
  o.thaw(label);
}

template<class T, class F>
void thaw(Array<T,F>& o, LazyLabel* label) {
  if (!is_value<T>::value) {
    auto iter = o.begin();
    auto last = iter + o.size();
    for (; iter != last; ++iter) {
      thaw(*iter, label);
    }
  }
}

template<class T>
void thaw(Optional<T>& o, LazyLabel* label) {
  if (!is_value<T>::value && o.query()) {
    thaw(o.get(), label);
  }
}

template<class T>
void thaw(std::function<T>& o) {
  assert(false);
  /// @todo Need to thaw any objects in the closure here, which may require
  /// a custom implementation of lambda functions in a similar way to fibers,
  /// rather than using std::function
}

template<class... Args>
void thaw(Tuple<Args...>& o, LazyLabel* label) {
  o.thaw(label);
}

template<class T>
void finish(Shared<T>& o) {
  o.finish();
}

template<class T>
void finish(Weak<T>& o) {
  o.finish();
}

template<class T>
void finish(Init<T>& o) {
  o.finish();
}

template<class T>
void finish(Fiber<T>& o) {
  o.finish();
}

template<class T, class F>
void finish(Array<T,F>& o) {
  if (!is_value<T>::value) {
    auto iter = o.begin();
    auto last = iter + o.size();
    for (; iter != last; ++iter) {
      finish(*iter);
    }
  }
}

template<class T>
void finish(Optional<T>& o) {
  if (!is_value<T>::value && o.query()) {
    finish(o.get());
  }
}

template<class T>
void finish(std::function<T>& o) {
  assert(false);
  /// @todo Need to finish any objects in the closure here, which may require
  /// a custom implementation of lambda functions in a similar way to fibers,
  /// rather than using std::function
}

template<class... Args>
void finish(Tuple<Args...>& o) {
  o.finish();
}
}
