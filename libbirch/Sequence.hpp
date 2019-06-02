/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"

namespace libbirch {
/**
 * Sequence type.
 *
 * @param Type Element type.
 * @param depth Number of dimensions.
 */
template<class Type, int depth>
struct sequence_type {
  using type = std::initializer_list<typename sequence_type<Type,depth - 1>::type>;
};

/**
 * Sequence type base case.
 */
template<class Type>
struct sequence_type<Type,1> {
  using type = std::initializer_list<Type>;
};

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
void sequence_lengths(const std::initializer_list<Type>& o,
    int64_t* lengths) {
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
 * Copy from a sequence into an array.
 */
template<class Iterator, class Type>
void sequence_copy(Iterator& to, const Type& from) {
  new (to.get()) typename Iterator::value_type(from);
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
