/**
 * @file
 */
#pragma once

#include "bi/visitor/Visitor.hpp"

#include <list>

namespace bi {
/**
 * Gathers formal parameters and arguments for a function call.
 *
 * @ingroup compiler_visitor
 */
template<class T>
class Gatherer: public Visitor {
public:
  /**
   * Destructor.
   */
  virtual ~Gatherer();

  virtual void visit(const T* o);

  /**
   * Parameters.
   */
  std::list<const T*> gathered;
};
}

template<class T>
bi::Gatherer<T>::~Gatherer() {
  //
}

template<class T>
void bi::Gatherer<T>::visit(const T* o) {
  gathered.push_back(o);
}
