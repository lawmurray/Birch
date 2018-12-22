/**
 * @file
 */
#include "bi/primitive/poset.hpp"

#include "bi/type/Type.hpp"
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
    roots.insert(v);
  }
  for (auto child : children1) {
    roots.erase(child);
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

template class bi::poset<bi::Type*,bi::definitely>;
template class bi::poset<bi::Unknown*,bi::definitely>;
template class bi::poset<bi::Function*,bi::definitely>;
template class bi::poset<bi::Fiber*,bi::definitely>;
template class bi::poset<bi::MemberFunction*,bi::definitely>;
template class bi::poset<bi::MemberFiber*,bi::definitely>;
template class bi::poset<bi::BinaryOperator*,bi::definitely>;
template class bi::poset<bi::UnaryOperator*,bi::definitely>;
