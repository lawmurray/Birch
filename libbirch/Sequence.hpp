/**
 * @file
 */
#pragma once

#include <initializer_list>

namespace bi {
/**
 * Depth of a sequence.
 */
template<class Type>
struct sequence_depth {
  static const int value = 0;
};
template<class Type>
struct sequence_depth<std::initializer_list<Type>> {
  static const int value = 1 + sequence_depth<Type>::value;
};

/**
 * Collect the lengths of nested sequences into an array.
 */
template<class Type>
void sequence_lengths(const Type& o, int64_t* lengths) {
  //
}
template<class Type>
void sequence_lengths(const std::initializer_list<Type>& o, int64_t* lengths) {
  *lengths = o.size();
  sequence_lengths(*o.begin(), lengths + 1);
}

/**
 * Create an appropriate frame for an array to be constructed from a sequence.
 */
template<class Type>
auto sequence_frame(const std::initializer_list<Type>& o) {
  typename DefaultFrame<sequence_depth<std::initializer_list<Type>>::value>::type frame;
  int64_t lengths[sequence_depth<std::initializer_list<Type>>::value];
  sequence_lengths(o, lengths);
  frame.resize(lengths);
  return frame;
}

/**
 * Does the shape of a sequence conform with that of the frame of an array?
 */
template<class Type>
bool sequence_conforms(const int64_t* sizes, const Type& o) {
  return true;
}
template<class Type>
bool sequence_conforms(const int64_t* sizes, const std::initializer_list<Type>& o) {
  if (*sizes != o.size()) {
    return false;
  }
  for (auto o1: o) {
    if (!sequence_conforms(sizes + 1, o1)) {
      return false;
    }
  }
  return true;
}

/**
 * Copy from a sequence into an array.
 */
template<class Iterator, class Type>
void sequence_copy(Iterator& to, const Type& from) {
  new (&*to) Type(from);
  ++to;
}
template<class Iterator, class Type>
void sequence_copy(Iterator& to, const std::initializer_list<Type>& from) {
  for (auto o : from) {
    sequence_copy(to, o);
  }
}

/**
 * Assign from a sequence into an array.
 */
template<class Iterator, class Type>
void sequence_assign(Iterator& to, const Type& from) {
  *to = from;
  ++to;
}
template<class Iterator, class Type>
void sequence_assign(Iterator& to, const std::initializer_list<Type>& from) {
  for (auto o : from) {
    sequence_assign(to, o);
  }
}

}
