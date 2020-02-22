/**
 * @file
 */
#include "bi/io/cpp/CppResumeGenerator.hpp"

#include "bi/visitor/Gatherer.hpp"
#include "bi/primitive/encode.hpp"

bi::CppResumeGenerator::CppResumeGenerator(const Fiber* currentFiber,
    std::ostream& base, const int level, const bool header) :
    CppBaseGenerator(base, level, header),
    currentFiber(currentFiber) {
  //
}

void bi::CppResumeGenerator::visit(const Function* o) {
  auto fiberType = dynamic_cast<const FiberType*>(o->returnType);
  assert(fiberType);
  auto generic = o->typeParams->width() + o->params->width() > 0;

  genSourceLine(o->loc);
  if (generic) {
    start("template<");
  }
  for (auto typeParam : *o->typeParams) {
    middle("class " << typeParam << ',');
  }
  if (!o->params->isEmpty()) {
    middle("class... Args");
  }
  if (generic) {
    finish('>');
  }

  if (header) {
    genSourceLine(o->loc);
    start("struct ");
    genUniqueName(o);
    middle(" final : public libbirch::FiberState<" << fiberType->yieldType << ',');
    finish(fiberType->returnType << "> {");
    in();

    if (!o->params->isEmpty()) {
      genSourceLine(o->loc);
      line("libbirch::Tuple<Args...> state_;\n");

      genSourceLine(o->loc);
      start("");
      genUniqueName(o);
      finish("(Args... args) : state_(args...) {");
      in();
      line("//");
      out();
      line("}\n");
    }

    genSourceLine(o->loc);
    line("virtual " << fiberType << " query();");
    start("LIBBIRCH_CLASS(");
    genUniqueName(o);
    middle(", libbirch::FiberState<" << fiberType->yieldType << ',');
    finish(fiberType->returnType << ">)");
    if (!o->params->isEmpty()) {
      line("LIBBIRCH_MEMBERS(state_)");
    } else {
      line("LIBBIRCH_MEMBERS()");
    }
    out();
    line("};\n");
  } else {
    genSourceLine(o->loc);
    start(fiberType << " bi::");
    genUniqueName(o);


    if (generic) {
      start('<');
    }
    for (auto typeParam : *o->typeParams) {
      middle("class " << typeParam << ',');
    }
    if (!o->params->isEmpty()) {
      middle("Args...");
    }
    if (generic) {
      middle('>');
    }
    finish("::query() {");
    in();
    genTraceFunction(o->name->str(), o->loc);
    genUnpack(o->params);
    *this << o->braces->strip();
    out();
    line("}");
  }
}

void bi::CppResumeGenerator::visit(const Yield* o) {
  auto resume = dynamic_cast<const Function*>(o->resume);
  assert(resume);

  genTraceLine(o->loc);
  start("return " << currentFiber->returnType << '(');
  if (!o->single->isEmpty()) {
    middle(o->single);
    if (!resume->params->isEmpty()) {
      middle(", ");
    }
  }
  if (!resume->params->isEmpty()) {
    middle("new ");
    genUniqueName(o);
    if (!currentFiber->typeParams->isEmpty()) {
      middle('<' << currentFiber->typeParams << '>');
    }
    middle('(');
    genPack(resume->params);
    middle(')');
  }
  finish(");");
}

void bi::CppResumeGenerator::visit(const Return* o) {
  genTraceLine(o->loc);
  start("return " << currentFiber->returnType << '(');
  if (!o->single->isEmpty()) {
    middle(o->single);
  }
  finish(");");
}

void bi::CppResumeGenerator::genUniqueName(const Numbered* o) {
  middle(currentFiber->name << '_' << currentFiber->number << '_');
  middle(o->number << '_');
}

void bi::CppResumeGenerator::genPack(const Expression* params) {
  for (auto iter = params->begin(); iter != params->end(); ++iter) {
    auto param = dynamic_cast<const Parameter*>(*iter);
    assert(param);
    if (iter != params->begin()) {
      middle(", ");
    }
    middle(getName(param->name->str(), param->number));
  }
}

void bi::CppResumeGenerator::genUnpack(const Expression* params) {
  auto i = 0;
  for (auto iter = params->begin(); iter != params->end(); ++iter) {
    auto param = dynamic_cast<const Parameter*>(*iter);
    assert(param);
    genSourceLine(param->loc);
    start("auto " << getName(param->name->str(), param->number));
    finish(" = state_.template get<" << i++ << ">();");
  }
}

std::string bi::CppResumeGenerator::getName(const std::string& name,
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
