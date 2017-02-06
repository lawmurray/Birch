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
   * Constructor.
   *
   * @predicate Optional predicate function to filter objects of type T.
   */
  Gatherer(std::function<bool(const T*)> predicate =
      [](const T* o) -> bool {return true;});

  /**
   * Destructor.
   */
  virtual ~Gatherer();

  /**
   * Begin iterator over gathered objects.
   */
  auto begin() {
    return gathered.begin();
  }

  /**
   * End iterator over gathered objects.
   */
  auto end() {
    return gathered.end();
  }

  /**
   * Number of items gathered.
   */
  auto size() {
    return gathered.size();
  }

  virtual void visit(const T* o);

protected:
  /**
   * Predicate.
   */
  std::function<bool(const T*)> predicate;

  /**
   * Gathered objects.
   */
  std::list<const T*> gathered;
};
}

template<class T>
bi::Gatherer<T>::Gatherer(std::function<bool(const T*)> predicate) :
    predicate(predicate) {
  //
}

template<class T>
bi::Gatherer<T>::~Gatherer() {
  //
}

template<class T>
void bi::Gatherer<T>::visit(const T* o) {
  if (predicate(o)) {
    gathered.push_back(o);
  }
}
