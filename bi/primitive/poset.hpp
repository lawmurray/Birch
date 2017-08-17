/**
 * @file
 */
#pragma once

#include <cstdlib>
#include <cstddef>
#include <algorithm>
#include <map>
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
   * Number of vertices in the poset.
   */
  auto size() const {
    return vertices.size();
  }

  /**
   * Does the set contain this value?
   */
  bool contains(T v) const;

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
   * Find all matches.
   *
   * @tparam Comparable Type comparable to value type.
   * @tparam Container Container type with push_back() function.
   *
   * @param v The value.
   * @param[out] matches Container to hold matches. Matches are inserted in
   * topological order, more specific first.
   */
  template<class Comparable, class Container>
  void match_all(Comparable v, Container& matches);

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

  /**
   * Sub-operation for match_all.
   */
  template<class Comparable, class Container>
  void match_all(T u, Comparable v, Container& matches);

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

#include <iostream>
#include <cassert>

template<class T, class Compare>
bi::poset<T,Compare>::poset() :
    colour(0) {
  //
}

template<class T, class Compare>
bool bi::poset<T,Compare>::contains(T v) const {
  return std::find(vertices.begin(), vertices.end(), v) != vertices.end();
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
void bi::poset<T,Compare>::match_all(Comparable v, Container& matches) {
  matches.clear();
  ++colour;
  for (auto iter = rbegin(); iter != rend(); ++iter) {
    match_all(*iter, v, matches);
  }
}

template<class T, class Compare>
void bi::poset<T,Compare>::insert(T v) {
  /* pre-condition */
  assert(!contains(v));

  add_colour(v);
  forward(v);
  backward(v);
  reduce();
  add_vertex(v);
}

template<class T, class Compare>
void bi::poset<T,Compare>::add_colour(T v) {
  colours.insert(std::make_pair(v, colour));
}

template<class T, class Compare>
void bi::poset<T,Compare>::add_vertex(T v) {
  /* determine location to insert so as to preserve topological order of
   * vertices */
  std::list<T> parents1, children1;
  parents(v, parents1);
  children(v, children1);
  std::ptrdiff_t l = 0, u = size();

  for (auto iter = parents1.begin(); iter != parents1.end(); ++iter) {
    auto find = std::find(begin(), end(), *iter);
    assert(find != end());
    u = std::min(u, std::distance(begin(), find));
  }
  for (auto iter = children1.begin(); iter != children1.end(); ++iter) {
    auto find = std::find(begin(), end(), *iter);
    assert(find != end());
    l = std::max(l, std::distance(begin(), find));
  }
  assert((l == 0 && u == 0) || l < u);

  /* insert */
  auto iter = begin();
  std::advance(iter, u);
  vertices.insert(iter, v);
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
        matches.push_back(u);
        deeper = true;
      }
    }
  }
  return deeper;
}

template<class T, class Compare>
template<class Comparable, class Container>
void bi::poset<T,Compare>::match_all(T u, Comparable v, Container& matches) {
  if (colours[u] < colour) {
    /* not visited yet */
    colours[u] = colour;
    if (compare(v, u)) {
      auto range = forwards.equal_range(u);
      for (auto iter = range.first; iter != range.second; ++iter) {
        match_all(iter->second, v, matches);
      }
      matches.push_back(u);
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
  for (auto iter = begin(); iter != end(); ++iter) {
    reduce(*iter);
  }
}

template<class T, class Compare>
void bi::poset<T,Compare>::reduce(T u) {
  int colour1 = ++colour;

  /* local copy of forward edges, as may change */
  std::list<T> forwards1;
  children(u, forwards1);

  /* depth first search discovery */
  for (auto iter = forwards1.begin(); iter != forwards1.end(); ++iter) {
    if (colours[*iter] < colour1) {
      colours[*iter] = colour1;
    }
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
void bi::poset<T,Compare>::dot() {
  ++colour;
  std::cout << "digraph {" << std::endl;
  for (auto iter = vertices.begin(); iter != vertices.end(); ++iter) {
    std::cout << "  \"" << (*iter)->name->str() << "\"" << std::endl;
  }
  for (auto iter = forwards.begin(); iter != forwards.end(); ++iter) {
    std::cout << "  \"" << iter->first->name->str() << "\" -> \""
        << iter->second->name->str() << "\"" << std::endl;
  }
  std::cout << "}" << std::endl;
}
