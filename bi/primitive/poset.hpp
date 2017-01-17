/**
 * @file
 */
#pragma once

#include "bi/primitive/possibly.hpp"

#include <cstdlib>
#include <vector>
#include <set>
#include <list>

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
   * Number of values in the poset.
   */
  size_t size() const;

  /**
   * Does the set contain a given value?
   */
  bool contains(T& value);

  /**
   * Get a given value.
   */
  T get(T& value);

  /**
   * Find the most-specific definite match(es) as well as more-specific
   * possible matche(es).
   *
   * @tparam Comparable Type comparable to value type.
   * @tparam Container Container type with push_back() function.
   *
   * @param value The value.
   * @param[out] definites Container to hold most-specific definite
   * match(es).
   * @param[out] possibles Container to hold more-specific possible
   * match(es).
   */
  template<class Comparable, class Container>
  void match(const Comparable& value, Container& definites,
      Container& possibles);

  /**
   * Insert vertex.
   *
   * @param value Value at the vertex.
   *
   * If the value equals that of another vertex, under the partial order
   * given, that vertex is overwritten.
   */
  void insert(T& value);

  /**
   * Output dot graph. Useful for diagnostic purposes.
   */
  void dot();

private:
  typedef std::set<int> set_type;

  /**
   * Add vertex.
   *
   * @param value Value at the vertex.
   *
   * @return Index of the vertex.
   */
  int add_vertex(const T& value);

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

  /**
   * Sub-operation for match to find most-specific definite match.
   *
   * @return True if a definite match was made in the subgraph.
   */
  template<class Comparable, class Container>
  bool match_definites(const int u, const Comparable& value,
      Container& definites);

  /**
   * Sub-operation for match to find most-specific definite match.
   */
  template<class Comparable, class Container>
  void match_possibles(const int u, const Comparable& value,
      Container& possibles);

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
  std::vector<T> values;

  /**
   * Vertex colours.
   */
  std::vector<int> colours;

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
  int colour;
};
}

#include <iostream>
#include <cassert>

template<class T, class Compare>
bi::poset<T,Compare>::poset() :
    colour(0) {
  //
}

template<class T, class Compare>
size_t bi::poset<T,Compare>::size() const {
  return values.size();
}

template<class T, class Compare>
bool bi::poset<T,Compare>::contains(T& value) {
  std::list<T> definites, possibles;
  match(value, definites, possibles);
  return definites.size() == 1
      && compare(definites.front(), value) == definite
      && compare(value, definites.front()) == definite;
}

template<class T, class Compare>
T bi::poset<T,Compare>::get(T& value) {
  /* pre-condition */
  assert(contains(value));

  std::list<T> definites, possibles;
  match(value, definites, possibles);

  return definites.front();
}

template<class T, class Compare>
template<class Comparable, class Container>
void bi::poset<T,Compare>::match(const Comparable& value,
    Container& definites, Container& possibles) {
  ++colour;
  for (auto iter = roots.begin(); iter != roots.end(); ++iter) {
    match_definites(*iter, value, definites);
  }

  ++colour;
  for (auto iter = roots.begin(); iter != roots.end(); ++iter) {
    match_possibles(*iter, value, possibles);
  }
}

template<class T, class Compare>
void bi::poset<T,Compare>::insert(T& value) {
  /* pre-condition */
  assert(!contains(value));

  const int v = add_vertex(value);
  forward(v);
  backward(v);
  reduce();
}

template<class T, class Compare>
int bi::poset<T,Compare>::add_vertex(const T& value) {
  const int v = values.size();

  values.push_back(value);
  forwards.push_back(set_type());
  backwards.push_back(set_type());
  colours.push_back(colour);
  roots.insert(v);
  leaves.insert(v);

  /* post-condition */
  assert(values.size() == forwards.size());
  assert(values.size() == backwards.size());
  assert(values.size() == colours.size());

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
bool bi::poset<T,Compare>::match_definites(const int u,
    const Comparable& value, Container& definites) {
  bool deeper = false;
  if (colours[u] < colour) {
    /* not visited yet */
    colours[u] = colour;
    if (compare(value, values[u]) == definite) {
      /* this vertex matcher, check if any vertices in the subgraph match
       * more-specifically */
      for (auto iter = forwards[u].begin(); iter != forwards[u].end();
          ++iter) {
        deeper = deeper || match_definites(*iter, value, definites);
      }
      if (!deeper) {
        /* no more-specific matches in the subgraph beneath this vertex, so
         * this is the most-specific match */
        definites.push_back(values[u]);
        deeper = true;
      }
    }
  }
  return deeper;
}

template<class T, class Compare>
template<class Comparable, class Container>
void bi::poset<T,Compare>::match_possibles(const int u,
    const Comparable& value, Container& possibles) {
  if (colours[u] < colour) {
    /* not visited yet */
    colours[u] = colour;
    possibly result = compare(value, values[u]);
    if (result != untrue) {
      /* either a definite or possible match, so continue searching through
       * the subgraph */
      for (auto iter = forwards[u].begin(); iter != forwards[u].end();
          ++iter) {
        match_possibles(*iter, value, possibles);
      }
    }
    if (result == possible) {
      /* if this was a possible (but not a definite) match, insert it in the
       * output; note this is done after searching the subgraph, so that
       * more-specific possible matches appear early in the output */
      possibles.push_back(values[u]);
    }
  }
}

template<class T, class Compare>
void bi::poset<T,Compare>::forward(const int v) {
  ++colour;
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
  if (colours[u] < colour) {
    colours[u] = colour;
    if (compare(values[u], values[v]) == definite) {
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
  ++colour;
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
  if (colours[u] < colour) {
    colours[u] = colour;
    if (compare(values[v], values[u]) == definite) {
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
  set_type lroots(roots);  // local copy as may change
  auto iter = lroots.begin();
  while (iter != lroots.end()) {
    reduce(*iter);
    ++iter;
  }
}

template<class T, class Compare>
void bi::poset<T,Compare>::reduce(const int u) {
  int lcolour = ++colour;
  set_type lforwards(forwards[u]);

  /* depth first search discovery */
  auto iter = lforwards.begin();
  while (iter != lforwards.end()) {
    if (colours[*iter] < lcolour) {
      colours[*iter] = lcolour;
    }
    reduce(*iter);
    ++iter;
  }

  /* remove edges for children that were rediscovered */
  iter = lforwards.begin();
  while (iter != lforwards.end()) {
    if (colours[*iter] > lcolour) {  // rediscovered
      remove_edge(u, *iter);
    }
    ++iter;
  }
}

template<class T, class Compare>
void bi::poset<T,Compare>::dot() {
  ++colour;
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
  if (colours[u] != colour) {
    colours[u] = colour;
    std::cout << "\"" << values[u] << "\"" << std::endl;
    auto iter = forwards[u].begin();
    while (iter != forwards[u].end()) {
      std::cout << "\"" << values[u] << "\" -> \"" << values[*iter] << "\""
          << std::endl;
      dot(*iter);
      ++iter;
    }
  }
}
