/**
 * @file
 */
#include "bi/io/cpp/CppFiberGenerator.hpp"

#include "bi/visitor/Gatherer.hpp"
#include "bi/primitive/encode.hpp"

bi::CppFiberGenerator::CppFiberGenerator(std::ostream& base,
    const int level, const bool header) :
    CppBaseGenerator(base, level, header),
    currentFiber(nullptr) {
  //
}

void bi::CppFiberGenerator::visit(const Fiber* o) {
  currentFiber = o;

  /* initialization function */
  genTemplateParams(o);
  genSourceLine(o->loc);
  start(o->returnType << ' ');
  if (!header) {
    middle("bi::");
  }
  middle(o->name << '(' << o->params << ')');
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();
    genTraceLine(o->loc);
    start("return make_fiber_" << o->name << "_0_(");
    middle("libbirch::make_tuple(");
    for (auto iter = o->params->begin(); iter != o->params->end(); ++iter) {
      if (iter != o->params->begin()) {
        middle(", ");
      }
      auto param = dynamic_cast<const Parameter*>(*iter);
      assert(param);
      middle(param->name);
    }
    finish("));");
    out();
    line("}\n");
  }

  /* start function */
  *this << o->start;

  /* resume functions */
  Gatherer<Yield> yields;
  o->accept(&yields);
  for (auto yield : yields) {
    if (yield->resume) {
      *this << yield->resume;
    }
  }
}

void bi::CppFiberGenerator::visit(const Function* o) {
  if (!o->braces->isEmpty()) {
    if (header) {
      /* struct with state and resume member function */
      genSourceLine(o->loc);
      start("template<");
      for (auto typeParam : *o->typeParams) {
        middle("class " << typeParam << ',');
      }
      finish("class State_>");
      genSourceLine(o->loc);
      line("struct " << o->name << '_' << o->number << "_ {");
      in();
      genSourceLine(o->loc);
      line("State_ state_;");
      genSourceLine(o->loc);
      line(o->returnType << " operator()();");
      out();
      line("};\n");

      /* factory function */
      genSourceLine(o->loc);
      start("template<");
      for (auto typeParam : *o->typeParams) {
        middle("class " << typeParam << ',');
      }
      finish("class Yield_, class State_>");
      genSourceLine(o->loc);
      start(o->returnType << " make_fiber_" << o->name << '_' << o->number << '_');
      finish("(Yield_&& yield_, State_&& state_) {");
      in();
      line("return " << o->returnType << "(yield_, " << o->name << '_' << o->number << "_<State_>{state_});");
      out();
      finish("}\n");
    } else {
      genSourceLine(o->loc);
      start("template<");
      for (auto typeParam : *o->typeParams) {
        middle("class " << typeParam << ',');
      }
      finish("class State_>");
      genSourceLine(o->loc);
      start(o->returnType << " bi::" << o->name << '_' << o->number << "_<");
      if (!o->typeParams->isEmpty()) {
        middle(o->typeParams << ',');
      }
      finish("State_>::operator()() {");
      in();
      genTraceFunction(o->name->str(), o->loc);
      genUnpack(o->params);
      *this << o->braces->strip();
      out();
      line("}\n");
    }
  }
}

void bi::CppFiberGenerator::visit(const Yield* o) {
  genTraceLine(o->loc);
  start("return make_fiber_" << currentFiber->name << '_' << o->number << "_(");
  middle(o->single << ", ");

  /* pack variables to be saved into state */
  auto resume = dynamic_cast<const Function*>(o->resume);
  assert(resume);
  genPack(resume->params);
  finish(");");
}

void bi::CppFiberGenerator::visit(const Return* o) {
  genTraceLine(o->loc);
  start("return libbirch::make_fiber<yield_type,return_type>");
  finish('(' << o->single << ");");
}

void bi::CppFiberGenerator::genPack(const Expression* params) {
  middle("libbirch::make_tuple(");
  for (auto iter = params->begin(); iter != params->end(); ++iter) {
    auto param = dynamic_cast<const Parameter*>(*iter);
    assert(param);
    if (iter != params->begin()) {
      middle(", ");
    }
    middle(getName(param->name->str(), param->number));
  }
  middle(')');
}

void bi::CppFiberGenerator::genUnpack(const Expression* params) {
  auto i = 0;
  for (auto iter = params->begin(); iter != params->end(); ++iter) {
    auto param = dynamic_cast<const Parameter*>(*iter);
    assert(param);
    genSourceLine(param->loc);
    start("auto " << getName(param->name->str(), param->number));
    finish(" = state_.template get<" << i++ << ">();");
  }
}

std::string bi::CppFiberGenerator::getName(const std::string& name,
    const int number) {
  std::stringstream buf;
  std::string result;
  auto iter = names.find(number);
  if (iter == names.end()) {
    auto count = counts.find(name);
    if (count == counts.end()) {
      buf << internalise(name);
      result = buf.str();
      counts.insert(std::make_pair(name, 1));
    } else {
      buf << internalise(name) << '_' << count->second << '_';
      result = buf.str();
      ++count->second;
    }
    names.insert(std::make_pair(number, result));
  } else {
    result = iter->second;
  }
  return result;
}
