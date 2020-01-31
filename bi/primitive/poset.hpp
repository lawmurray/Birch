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
   * Clear the container.
   */
  void clear();

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
   *
   * Use std::list, not std::set, to ensure that sort order is consistent
   * across runs.
   */
  std::list<T> roots;

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

#include "bi/io/bih_ostream.hpp"

template<class T, class Compare>
bi::poset<T,Compare>::poset() :
    colour(0) {
  //
}

template<class T, class Compare>
void bi::poset<T,Compare>::clear() {
  colour = 0;
  roots.clear();
  vertices.clear();
  forwards.clear();
  backwards.clear();
  colours.clear();
}

template<class T, class Compare>
bool bi::poset<T,Compare>::contains(T v) {
  std::set<T> matches;
  match(v, matches);

  /* are any of these an exact match? */
  for (auto u : matches) {
    if (compare(u, v) && compare(v, u)) {
      return true;
    }
  }
  return false;
}

template<class T, class Compare>
T bi::poset<T,Compare>::get(T v) {
  std::set<T> matches;
  match(v, matches);

  /* are any of these an exact match? */
  for (auto u : matches) {
    if (compare(u, v) && compare(v, u)) {
      return u;
    }
  }
  assert(false);
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

template<class T, class Compare>
void bi::poset<T,Compare>::insert(T v) {
  /* pre-condition */
  assert(!contains(v));

  add_colour(v);
  forward(v);
  backward(v);
  add_vertex(v);
  reduce();
  sort();
}

template<class T, class Compare>
void bi::poset<T,Compare>::add_colour(T v) {
  colours.insert(std::make_pair(v, colour));
}

template<class T, class Compare>
void bi::poset<T,Compare>::add_vertex(T v) {
  std::list<T> parents1, children1;
  parents(v, parents1);
  children(v, children1);

  if (parents1.empty()) {
    roots.push_back(v);
  }
  for (auto child : children1) {
    auto iter = std::find(roots.begin(), roots.end(), child);
    if (iter != roots.end()) {
      roots.erase(iter);
    }
  }
}

template<class T, class Compare>
void bi::poset<T,Compare>::add_edge(T u, T v) {
  /* pre-condition */
  assert(u != v);

  forwards.insert(std::make_pair(u, v));
  backwards.insert(std::make_pair(v, u));
}

template<class T, class Compare>
void bi::poset<T,Compare>::remove_edge(T u, T v) {
  /* remove forward edge */
  auto range1 = forwards.equal_range(u);
  bool found1 = false;
  for (auto iter1 = range1.first; !found1 && iter1 != range1.second;
      ++iter1) {
    if (iter1->second == v) {
      forwards.erase(iter1);
      found1 = true;
    }
  }

  /* remove backward edge */
  auto range2 = backwards.equal_range(v);
  bool found2 = false;
  for (auto iter2 = range2.first; !found2 && iter2 != range2.second;
      ++iter2) {
    if (iter2->second == u) {
      backwards.erase(iter2);
      found2 = true;
    }
  }
}

template<class T, class Compare>
void bi::poset<T,Compare>::forward(T v) {
  colours[v] = ++colour;
  for (auto iter = begin(); iter != end(); ++iter) {
    forward(*iter, v);
  }
}

template<class T, class Compare>
void bi::poset<T,Compare>::forward(T u, T v) {
  if (colours[u] < colour) {
    colours[u] = colour;
    if (compare(u, v) && !compare(v, u)) {
      add_edge(v, u);
    } else {
      std::list<T> forwards1;
      children(u, forwards1);
      for (auto iter = forwards1.begin(); iter != forwards1.end(); ++iter) {
        forward(*iter, v);
      }
    }
  }
}

template<class T, class Compare>
void bi::poset<T,Compare>::backward(T v) {
  colours[v] = ++colour;
  for (auto iter = rbegin(); iter != rend(); ++iter) {
    backward(*iter, v);
  }
}

template<class T, class Compare>
void bi::poset<T,Compare>::backward(T u, T v) {
  if (colours[u] < colour) {
    colours[u] = colour;
    if (compare(v, u) && !compare(u, v)) {
      add_edge(u, v);
    } else {
      std::list<T> backwards1;
      parents(u, backwards1);
      for (auto iter = backwards1.begin(); iter != backwards1.end(); ++iter) {
        backward(*iter, v);
      }
    }
  }
}

template<class T, class Compare>
void bi::poset<T,Compare>::reduce() {
  for (auto u: roots) {
    reduce(u);
  }
}

template<class T, class Compare>
void bi::poset<T,Compare>::reduce(T u) {
  int colour1 = ++colour;

  /* local copy of forward edges, as may change */
  std::list<T> forwards1;
  children(u, forwards1);

  /* colour children */
  for (auto iter = forwards1.begin(); iter != forwards1.end(); ++iter) {
    if (colours[*iter] < colour1) {
      colours[*iter] = colour1;
    }
  }

  /* depth first search discovery */
  for (auto iter = forwards1.begin(); iter != forwards1.end(); ++iter) {
    reduce(*iter);
  }

  /* remove edges for children that were rediscovered */
  for (auto iter = forwards1.begin(); iter != forwards1.end(); ++iter) {
    if (colours[*iter] > colour1) {  // rediscovered
      remove_edge(u, *iter);
    }
  }
}

template<class T, class Compare>
void bi::poset<T,Compare>::sort() {
  ++colour;
  vertices.clear();
  for (auto u : roots) {
    sort(u);
  }
}

template<class T, class Compare>
void bi::poset<T,Compare>::sort(T u) {
  if (colours[u] < colour) {
    colours[u] = colour;
    vertices.push_front(u);

    std::list<T> children1;
    children(u, children1);
    for (auto v : children1) {
      sort(v);
    }
  }
}

template<class T, class Compare>
void bi::poset<T,Compare>::dot() {
  bih_ostream buf(std::cerr);
  buf << "digraph {\n";
  for (auto iter = vertices.begin(); iter != vertices.end(); ++iter) {
    buf << "  \"" << *iter << "\"\n";
  }
  for (auto iter = forwards.begin(); iter != forwards.end(); ++iter) {
    buf << "  \"" << iter->first << "\" -> \"" << iter->second << "\"\n";
  }
  buf << "}\n";
}

template<class T, class Compare>
template<class Container>
void bi::poset<T,Compare>::children(T v, Container& children) {
  auto range = forwards.equal_range(v);
  for (auto iter = range.first; iter != range.second; ++iter) {
    children.push_back(iter->second);
  }
}
