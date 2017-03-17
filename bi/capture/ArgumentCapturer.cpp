/**
 * @file
 */
#include "bi/capture/ArgumentCapturer.hpp"

#include "bi/expression/FuncReference.hpp"
#include "bi/expression/FuncParameter.hpp"
#include "bi/dispatcher/Dispatcher.hpp"
#include "bi/visitor/Gatherer.hpp"

bi::ArgumentCapturer::ArgumentCapturer(const FuncReference* ref,
    const FuncParameter* param) {
  FuncReference* ref1 = const_cast<FuncReference*>(ref);
  FuncParameter* param1 = const_cast<FuncParameter*>(param);

  bool result = ref1->definitely(*param1);
  assert(result);

  Gatherer<VarParameter> gatherer;
  param1->parens->accept(&gatherer);

  for (auto iter = gatherer.begin(); iter != gatherer.end(); ++iter) {
    gathered.push_back(std::make_pair((*iter)->arg, *iter));
  }
}

bi::ArgumentCapturer::ArgumentCapturer(const FuncReference* ref,
    const Dispatcher* param) {
  FuncReference* ref1 = const_cast<FuncReference*>(ref);
  Dispatcher* param1 = const_cast<Dispatcher*>(param);

  bool result = ref1->parens->possibly(*param1->parens);
  assert(result);

  Gatherer<VarParameter> gatherer;
  param1->parens->accept(&gatherer);

  for (auto iter = gatherer.begin(); iter != gatherer.end(); ++iter) {
    gathered.push_back(std::make_pair((*iter)->arg, *iter));
  }
}

bi::ArgumentCapturer::ArgumentCapturer(const Expression* parens1,
    const Expression* parens2) {
  Expression* ref1 = const_cast<Expression*>(parens1);
  Expression* param1 = const_cast<Expression*>(parens2);

  bool result = ref1->possibly(*param1);
  assert(result);

  Gatherer<VarParameter> gatherer;
  param1->accept(&gatherer);

  for (auto iter = gatherer.begin(); iter != gatherer.end(); ++iter) {
    gathered.push_back(std::make_pair((*iter)->arg, *iter));
  }
}
