/**
 * @file
 */
#pragma once

namespace bi {
/**
 * Partially ordered set.
 *
 * @tparam T Value type.
 * @tparam Compare Comparison functor.
 */
template<class T, class Compare>
class poset {
public:
  /**
   * Constructor.
   */
  poset();

  /**
   * Number of vertices in the poset.
   */
  auto size() const {
    return vertices.size();
  }

  /**
   * Does the set contain this value?
   */
  bool contains(T v);

  /**
   * Get the value, if the set contains it.
   */
  T get(T v);

  /**
   * Get the children of a vertex.
   */
  template<class Container>
  void children(T v, Container& children);

  /**
   * Get the parents of a vertex.
   */
  template<class Container>
  void parents(T v, Container& parents);

  /**
   * Find the most-specific match(es).
   *
   * @tparam Comparable Type comparable to value type.
   * @tparam Container Container type with push_back() function.
   *
   * @param v The value.
   * @param[out] matches Container to hold matches.
   *
   * @p matches may contain duplicates if the container type permits. It is
   * recommended that a container that does not permit duplicates, such as
   * std::set.
   */
  template<class Comparable, class Container>
  void match(Comparable v, Container& matches);

  /**
   * Insert vertex.
   *
   * @param v Value at the vertex.
   */
  void insert(T v);

  /*
   * Iterators over vertices. These iterate over vertices in a valid
   * topological order. More specific first.
   */
  auto begin() {
    return vertices.begin();
  }
  auto end() {
    return vertices.end();
  }
  auto begin() const {
    return vertices.begin();
  }
  auto end() const {
    return vertices.end();
  }
  auto rbegin() {
    return vertices.rbegin();
  }
  auto rend() {
    return vertices.rend();
  }
  auto rbegin() const {
    return vertices.rbegin();
  }
  auto rend() const {
    return vertices.rend();
  }

  /**
   * Output dot graph. Useful for diagnostic purposes.
   */
  void dot();

private:
  /**
   * Add colour.
   *
   * @param v Vertex.
   */
  void add_colour(T v);

  /**
   * Add vertex.
   *
   * @param v Vertex.
   */
  void add_vertex(T v);

  /**
   * Add edge.
   *
   * @param u Source vertex index.
   * @param v Destination vertex index.
   */
  void add_edge(T u, T v);

  /**
   * Remove edge.
   *
   * @param u Source vertex index.
   * @param v Destination vertex index.
   */
  void remove_edge(T u, T v);

  /**
   * Sub-operation for match.
   */
  template<class Comparable, class Container>
  bool match(T u, Comparable v, Container& matches);

  /*
   * Sub-operations for insert.
   */
  void forward(T v);
  void forward(T u, T v);
  void backward(T v);
  void backward(T u, T v);
  void reduce();  // transitive reduction
  void reduce(T u);

  /**
   * Sort vertices in topological order into vertices.
   */
  void sort();
  void sort(T u);

  /**
   * Root vertices.
   */
  std::set<T> roots;

  /**
   * Vertices in topological order.
   */
  std::list<T> vertices;

  /**
   * Forward and backward edges.
   */
  std::multimap<T,T> forwards, backwards;

  /**
   * Vertex colours.
   */
  std::map<T,int> colours;

  /**
   * Comparison operator.
   */
  Compare compare;

  /**
   * Current colour.
   */
  int colour;
};
}

template<class T, class Compare>
template<class Container>
void bi::poset<T,Compare>::children(T v, Container& children) {
  auto range = forwards.equal_range(v);
  for (auto iter = range.first; iter != range.second; ++iter) {
    children.push_back(iter->second);
  }
}

template<class T, class Compare>
template<class Container>
void bi::poset<T,Compare>::parents(T v, Container& parents) {
  auto range = backwards.equal_range(v);
  for (auto iter = range.first; iter != range.second; ++iter) {
    parents.push_back(iter->second);
  }
}

template<class T, class Compare>
template<class Comparable, class Container>
void bi::poset<T,Compare>::match(Comparable v, Container& matches) {
  matches.clear();
  ++colour;
  for (auto iter = rbegin(); iter != rend(); ++iter) {
    match(*iter, v, matches);
  }
}

template<class T, class Compare>
template<class Comparable, class Container>
bool bi::poset<T,Compare>::match(T u, Comparable v, Container& matches) {
  bool deeper = false;
  if (colours[u] < colour) {
    /* not visited yet */
    colours[u] = colour;
    if (compare(v, u)) {
      /* this vertex matches, check if any vertices in the subgraph match
       * more-specifically */
      auto range = forwards.equal_range(u);
      for (auto iter = range.first; iter != range.second; ++iter) {
        deeper = match(iter->second, v, matches) || deeper;
        // ^ do the || in this order to prevent short circuit
      }
      if (!deeper) {
        /* no more-specific matches in the subgraph beneath this vertex, so
         * this is the most-specific match */
        matches.insert(u);
        deeper = true;
      }
    }
  } else {
    /* already visited */
    deeper = compare(v, u);
  }
  return deeper;
}
