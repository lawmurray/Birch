/**
 * @file
 */
#pragma once

#include <initializer_list>

namespace bi {
/**
 * Sequence.
 */
template<class Type>
using Sequence = std::initializer_list<Type>;

/**
 * Depth of a sequence.
 */
template<class Type>
struct sequence_depth {
  static const int value = 0;
};
template<class Type>
struct sequence_depth<Sequence<Type>> {
  static const int value = 1 + sequence_depth<Type>::value;
};

/**
 * Collect the lengths of nested sequences into an array.
 */
template<class Type>
void sequence_lengths(const Type& o, size_t* lengths) {
  //
}
template<class Type>
void sequence_lengths(const Sequence<Type>& o,
    size_t* lengths) {
  *lengths = o.size();
  sequence_lengths(*o.begin(), lengths + 1);
}

/**
 * Create an appropriate frame for an array to be constructed from a sequence.
 */
template<class Type>
auto sequence_frame(const Sequence<Type>& o) {
  size_t lengths[sequence_depth<Sequence<Type>>::value];
  sequence_lengths(o, lengths);

  typename DefaultFrame<sequence_depth<Sequence<Type>>::value>::type frame;
  frame.setLengths(lengths);

  return frame;
}

/**
 * Does the shape of a sequence conform with that of the frame of an array?
 */
template<class Type>
bool sequence_conforms(const size_t* sizes, const Type& o) {
  return true;
}
template<class Type>
bool sequence_conforms(const size_t* sizes, const Sequence<Type>& o) {
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
template<class Type, class Iterator>
void sequence_copy(const Type& from, Iterator& to) {
  *to = from;
  ++to;
}
template<class Type, class Iterator>
void sequence_copy(const Sequence<Type>& from, Iterator& to) {
  for (auto o : from) {
    sequence_copy(o, to);
  }
}

}
