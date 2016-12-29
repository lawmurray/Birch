/**
 * @file
 */
#pragma once

#include <cstdlib>
#include <vector>
#include <set>
#include <list>

namespace bi {
/**
 * Partially ordered set.
 *
 * @ingroup primitive_container
 *
 * @tparam T Value type.
 * @tparam Compare Comparison functor.
 */
template<class T, class Compare = std::less<T> >
class poset {
public:
  /**
   * Constructor.
   */
  poset();

  /**
   * Number of values in the poset.
   */
  size_t size() const;

  /**
   * For a singleton set, return the single element.
   */
  T& only();

  /**
   * Does the set contain a given value?
   */
  bool contains(T& val);

  /**
   * Get a given value.
   */
  T get(T& val);

  /**
   * Find what would be the parent vertices for a given value.
   *
   * @tparam Comparable Type comparable to value type.
   * @tparam Container Container type with push_back() function.
   *
   * @param val The value to find.
   * @param[out] out Container in which to insert values of parent vertices.
   */
  template<class Comparable, class Container>
  void find(const Comparable& val, Container& out);

  /**
   * Insert vertex.
   *
   * @param val Value at the vertex.
   *
   * If the value equals that of another vertex, under the partial order
   * given, that vertex is overwritten.
   */
  void insert(T& val);

  /**
   * Output dot graph. Useful for diagnostic purposes.
   */
  void dot();

private:
  typedef std::set<int> set_type;

  /**
   * Add vertex.
   *
   * @param val Value at the vertex.
   *
   * @return Index of the vertex.
   */
  int add_vertex(const T& val);

  /**
   * Add edge.
   *
   * @param u Source vertex index.
   * @param v Destination vertex index.
   */
  void add_edge(const int u, const int v);

  /**
   * Remove edge.
   *
   * @param u Source vertex index.
   * @param v Destination vertex index.
   */
  void remove_edge(const int y, const int v);

  /*
   * Sub-operations for find.
   */
  template<class Comparable, class Container>
  void find(const int u, const Comparable& val, Container& out);

  /*
   * Sub-operations for insert.
   */
  void forward(const int v);
  void forward(const int u, const int v);
  void backward(const int v);
  void backward(const int u, const int v);
  void reduce();  // transitive reduction
  void reduce(const int u);

  /*
   * Sub-operations for dot.
   */
  void dot(const int u);

  /**
   * Vertex values.
   */
  std::vector<T> vals;

  /**
   * Vertex colours.
   */
  std::vector<int> cols;

  /**
   * Forward and backward edges.
   */
  std::vector<set_type> forwards, backwards;

  /**
   * Roots and leaves.
   */
  set_type roots, leaves;

  /**
   * Comparison.
   */
  Compare compare;

  /**
   * Current colour.
   */
  int col;
};
}

#include <iostream>
#include <cassert>

template<class T, class Compare>
bi::poset<T,Compare>::poset() :
    col(0) {
  //
}

template<class T, class Compare>
size_t bi::poset<T,Compare>::size() const {
  return vals.size();
}

template<class T, class Compare>
T& bi::poset<T,Compare>::only() {
  /* pre-condition */
  assert(size() == 1);

  return vals.front();
}

template<class T, class Compare>
bool bi::poset<T,Compare>::contains(T& val) {
  std::list<T> out;
  find(val, out);
  return out.size() == 1 && compare(out.front(), val)
      && compare(val, out.front());
}

template<class T, class Compare>
T bi::poset<T,Compare>::get(T& val) {
  /* pre-condition */
  assert(contains(val));

  std::list<T> out;
  find(val, out);

  return out.front();
}

template<class T, class Compare>
template<class Comparable, class Container>
void bi::poset<T,Compare>::find(const Comparable& val, Container& out) {
  ++col;
  auto iter = roots.begin();
  while (iter != roots.end()) {
    cols[*iter] = col;
    if (compare(val, vals[*iter])) {
      find(*iter, val, out);
    }
    ++iter;
  }
}

template<class T, class Compare>
void bi::poset<T,Compare>::insert(T& val) {
  /* pre-condition */
  assert(!contains(val));

  const int v = add_vertex(val);
  forward(v);
  backward(v);
  reduce();
}

template<class T, class Compare>
int bi::poset<T,Compare>::add_vertex(const T& val) {
  const int v = vals.size();

  vals.push_back(val);
  forwards.push_back(set_type());
  backwards.push_back(set_type());
  cols.push_back(col);
  roots.insert(v);
  leaves.insert(v);

  /* post-condition */
  assert(vals.size() == forwards.size());
  assert(vals.size() == backwards.size());
  assert(vals.size() == cols.size());

  return v;
}

template<class T, class Compare>
void bi::poset<T,Compare>::add_edge(const int u, const int v) {
  if (u != v) {
    forwards[u].insert(v);
    backwards[v].insert(u);
    leaves.erase(u);
    roots.erase(v);
  }
}

template<class T, class Compare>
void bi::poset<T,Compare>::remove_edge(const int u, const int v) {
  forwards[u].erase(v);
  backwards[v].erase(u);
  if (forwards[u].size() == 0) {
    leaves.insert(u);
  }
  if (backwards[v].size() == 0) {
    roots.insert(v);
  }
}

template<class T, class Compare>
template<class Comparable, class Container>
void bi::poset<T,Compare>::find(const int u, const Comparable& val,
    Container& out) {
  bool deeper = false;
  auto iter = forwards[u].begin();
  while (iter != forwards[u].end()) {
    if (cols[*iter] < col) {
      cols[*iter] = col;
      if (compare(val, vals[*iter])) {
        deeper = true;
        find(*iter, val, out);
      }
    }
    ++iter;
  }
  if (!deeper) {
    out.push_back(vals[u]);
  }
}

template<class T, class Compare>
void bi::poset<T,Compare>::forward(const int v) {
  ++col;
  auto roots1 = roots;  // local copy as may be modified during iteration
  auto iter = roots1.begin();
  while (iter != roots1.end()) {
    if (*iter != v) {
      forward(*iter, v);
    }
    ++iter;
  }
}

template<class T, class Compare>
void bi::poset<T,Compare>::forward(const int u, const int v) {
  if (cols[u] < col) {
    cols[u] = col;
    if (compare(vals[u], vals[v])) {
      add_edge(v, u);
    } else {
      auto iter = forwards[u].begin();
      while (iter != forwards[u].end()) {
        forward(*iter, v);
        ++iter;
      }
    }
  }
}

template<class T, class Compare>
void bi::poset<T,Compare>::backward(const int v) {
  ++col;
  auto leaves1 = leaves;  // local copy as may be modified during iteration
  auto iter = leaves1.begin();
  while (iter != leaves1.end()) {
    if (*iter != v) {
      backward(*iter, v);
    }
    ++iter;
  }
}

template<class T, class Compare>
void bi::poset<T,Compare>::backward(const int u, const int v) {
  if (cols[u] < col) {
    cols[u] = col;
    if (compare(vals[v], vals[u])) {
      add_edge(u, v);
    } else {
      auto iter = backwards[u].begin();
      while (iter != backwards[u].end()) {
        backward(*iter, v);
        ++iter;
      }
    }
  }
}

template<class T, class Compare>
void bi::poset<T,Compare>::reduce() {
  set_type lroots(roots);
  auto iter = lroots.begin();
  while (iter != lroots.end()) {
    reduce(*iter);
    ++iter;
  }
}

template<class T, class Compare>
void bi::poset<T,Compare>::reduce(const int u) {
  int lcol = ++col;
  set_type lforwards(forwards[u]);

  /* depth first search discovery */
  auto iter = lforwards.begin();
  while (iter != lforwards.end()) {
    if (cols[*iter] < lcol) {
      cols[*iter] = lcol;
    }
    reduce(*iter);
    ++iter;
  }

  /* remove edges for children that were rediscovered */
  iter = lforwards.begin();
  while (iter != lforwards.end()) {
    if (cols[*iter] > lcol) {  // rediscovered
      remove_edge(u, *iter);
    }
    ++iter;
  }
}

template<class T, class Compare>
void bi::poset<T,Compare>::dot() {
  ++col;
  std::cout << "digraph {" << std::endl;
  auto iter = roots.begin();
  while (iter != roots.end()) {
    dot(*iter);
    ++iter;
  }
  std::cout << "}" << std::endl;
}

template<class T, class Compare>
void bi::poset<T,Compare>::dot(const int u) {
  if (cols[u] != col) {
    cols[u] = col;
    std::cout << "\"" << vals[u] << "\"" << std::endl;
    auto iter = forwards[u].begin();
    while (iter != forwards[u].end()) {
      std::cout << "\"" << vals[u] << "\" -> \"" << vals[*iter] << "\""
          << std::endl;
      dot(*iter);
      ++iter;
    }
  }
}
