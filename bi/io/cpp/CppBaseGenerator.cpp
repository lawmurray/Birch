/**
 * @file
 */
#include "bi/io/cpp/CppBaseGenerator.hpp"

#include "bi/io/cpp/CppClassGenerator.hpp"
#include "bi/io/cpp/CppFiberGenerator.hpp"
#include "bi/primitive/encode.hpp"

#include "boost/filesystem.hpp"
#include "boost/algorithm/string.hpp"

bi::CppBaseGenerator::CppBaseGenerator(std::ostream& base, const int level,
    const bool header) :
    indentable_ostream(base, level),
    header(header) {
  //
}

void bi::CppBaseGenerator::visit(const Name* o) {
  middle(internalise(o->str()));
}

void bi::CppBaseGenerator::visit(const ExpressionList* o) {
  middle(o->head);
  if (o->tail) {
    middle(", " << o->tail);
  }
}

void bi::CppBaseGenerator::visit(const Literal<bool>* o) {
  middle("bi::type::Boolean_(" << o->str << ")");
}

void bi::CppBaseGenerator::visit(const Literal<int64_t>* o) {
  middle("bi::type::Integer64_(" << o->str << ")");
}

void bi::CppBaseGenerator::visit(const Literal<double>* o) {
  middle("bi::type::Real64_(" << o->str << ")");
}

void bi::CppBaseGenerator::visit(const Literal<const char*>* o) {
  middle("std::string(" << o->str << ')');
}

void bi::CppBaseGenerator::visit(const Parentheses* o) {
  if (o->single->type->isList()) {
    if (o->single->isAssignable()) {
      middle("std::tie(" << o->single << ')');
    } else {
      middle("std::make_tuple(" << o->single << ')');
    }
  } else {
    middle('(' << o->single << ')');
  }
}

void bi::CppBaseGenerator::visit(const Sequence* o) {
  middle("bi::make_sequence({" << o->single << "})");
}

void bi::CppBaseGenerator::visit(const Cast* o) {
  auto classType = dynamic_cast<ClassType*>(o->returnType);
  if (o->single->type->isOptional()) {
    middle("[](auto o) -> auto { return o.is_initialized()? ");
    middle("o.get().template cast<bi::");
    middle(classType->name);
    middle(">(): boost::optional<bi::Pointer<bi::");
    middle(classType->name);
    middle(">>{}; }(");
    middle(o->single);
    middle(")");
  } else {
    middle(o->single);
    middle(".cast<bi::type::" << classType->name << ">()");
  }
}

void bi::CppBaseGenerator::visit(const Call* o) {
  middle(o->single);
  genArgs(o);
}

void bi::CppBaseGenerator::visit(const BinaryCall* o) {
  auto op = dynamic_cast<OverloadedIdentifier<BinaryOperator>*>(o->single);
  assert(op);
  if (isTranslatable(op->name->str())) {
    /* can use corresponding C++ operator */
    genLeftArg(o);
    middle(' ' << op->name->str() << ' ');
    genRightArg(o);
  } else {
    /* must use as function */
    middle(o->single << '(');
    genLeftArg(o);
    middle(", ");
    genRightArg(o);
    middle(')');
  }
}

void bi::CppBaseGenerator::visit(const UnaryCall* o) {
  auto op = dynamic_cast<OverloadedIdentifier<UnaryOperator>*>(o->single);
  assert(op);
  if (isTranslatable(op->name->str())) {
    /* can use corresponding C++ operator */
    middle(op->name->str());
    genSingleArg(o);
  } else {
    /* must use as function */
    middle(o->single << '(');
    genSingleArg(o);
    middle(')');
  }
}

void bi::CppBaseGenerator::visit(const Slice* o) {
  middle(o->single << "(bi::make_view(" << o->brackets << "))");
}

void bi::CppBaseGenerator::visit(const Query* o) {
  if (o->single->type->isOptional()) {
    middle("static_cast<bool>(" << o->single << ')');
  } else {
    middle(o->single << ".query()");
  }
}

void bi::CppBaseGenerator::visit(const Get* o) {
  middle(o->single << ".get()");
}

void bi::CppBaseGenerator::visit(const LambdaFunction* o) {
  finish("[=](" << o->params << ") {");
  in();
  *this << o->braces->strip();
  out();
  start("}");
}

void bi::CppBaseGenerator::visit(const Span* o) {
  middle("bi::make_span(");
  if (o->single->isEmpty()) {
    middle('0');
  } else {
    middle(o->single);
  }
  middle(')');
}

void bi::CppBaseGenerator::visit(const Index* o) {
  middle("bi::make_index(" << o->single << " - 1)");
}

void bi::CppBaseGenerator::visit(const Range* o) {
  middle("bi::make_range(" << o->left << " - 1, " << o->right << " - 1)");
}

void bi::CppBaseGenerator::visit(const Member* o) {
  const This* leftThis = dynamic_cast<const This*>(o->left);
  const Super* leftSuper = dynamic_cast<const Super*>(o->left);
  if (leftThis) {
    middle("this->");
  } else if (leftSuper) {
    middle("this->super_type::");
  } else {
    middle(o->left << "->");
  }
  middle(o->right);
}

void bi::CppBaseGenerator::visit(const Super* o) {
  // only need to handle the case outside member expression
  middle("pointer_from_this<super_type>()");
}

void bi::CppBaseGenerator::visit(const This* o) {
  // only need to handle the case outside member expression, where must
  // ensure correct global or fiber-local pointer is used
  middle("pointer_from_this<this_type>()");
}

void bi::CppBaseGenerator::visit(const Nil* o) {
  middle("boost::none");
}

void bi::CppBaseGenerator::visit(const Parameter* o) {
  middle("const " << o->type << "& " << o->name);
  if (!o->value->isEmpty()) {
    middle(" = " << o->value);
  }
}

void bi::CppBaseGenerator::visit(const MemberParameter* o) {
    middle("const " << o->type << "& " << o->name);
  if (!o->value->isEmpty()) {
    middle(" = " << o->value);
  }
}

void bi::CppBaseGenerator::visit(const Identifier<Parameter>* o) {
  middle(o->name);
}

void bi::CppBaseGenerator::visit(const Identifier<MemberParameter>* o) {
  middle(o->name);
}

void bi::CppBaseGenerator::visit(const Identifier<GlobalVariable>* o) {
  middle(o->name);
}

void bi::CppBaseGenerator::visit(const Identifier<LocalVariable>* o) {
  middle(o->name);
}

void bi::CppBaseGenerator::visit(const Identifier<MemberVariable>* o) {
  middle(o->name);
}

void bi::CppBaseGenerator::visit(const OverloadedIdentifier<Function>* o) {
  middle("bi::" << o->name);
}

void bi::CppBaseGenerator::visit(const OverloadedIdentifier<Fiber>* o) {
  middle("bi::" << o->name);
}

void bi::CppBaseGenerator::visit(
    const OverloadedIdentifier<MemberFunction>* o) {
  middle(o->name);
}

void bi::CppBaseGenerator::visit(const OverloadedIdentifier<MemberFiber>* o) {
  middle(o->name);
}

void bi::CppBaseGenerator::visit(
    const OverloadedIdentifier<BinaryOperator>* o) {
  middle(o->name);
}

void bi::CppBaseGenerator::visit(
    const OverloadedIdentifier<UnaryOperator>* o) {
  middle(o->name);
}

void bi::CppBaseGenerator::visit(const File* o) {
  line("/**");
  line(" * @file");
  line(" *");
  line(" * Automatically generated by Birch.");
  line(" */");

  if (header) {
    /* include guard */
    line("#pragma once\n");
  }

  /* main code */
  *this << o->root;
}

void bi::CppBaseGenerator::visit(const GlobalVariable* o) {
  if (header) {
    line("extern " << o->type << ' ' << o->name << ';');
  } else {
    start(o->type << " bi::" << o->name);
    genInit(o);
    finish(';');
  }
}

void bi::CppBaseGenerator::visit(const LocalVariable* o) {
  middle(o->type << ' ' << o->name);
  genInit(o);
}

void bi::CppBaseGenerator::visit(const MemberVariable* o) {
  assert(false);  // should be in CppClassGenerator
}

void bi::CppBaseGenerator::visit(const Function* o) {
  if (!o->braces->isEmpty()) {
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

      /* body */
      CppBaseGenerator aux(base, level, false);
      aux << o->braces->strip();

      out();
      finish("}\n");
    }
  }
}

void bi::CppBaseGenerator::visit(const Fiber* o) {
  CppFiberGenerator auxFiber(base, level, header);
  auxFiber << o;
}

void bi::CppBaseGenerator::visit(const MemberFunction* o) {
  assert(false);  // should be in CppClassGenerator
}

void bi::CppBaseGenerator::visit(const MemberFiber* o) {
  assert(false);  // should be in CppClassGenerator
}

void bi::CppBaseGenerator::visit(const Program* o) {
  if (header) {
    line("extern \"C\" void " << o->name << "(int argc, char** argv);");
  } else {
    line("void bi::" << o->name << "(int argc, char** argv) {");
    in();

    /* handle program options */
    if (o->params->width() > 0) {
      /* option variables */
      for (auto iter = o->params->begin(); iter != o->params->end(); ++iter) {
        auto param = dynamic_cast<const Parameter*>(*iter);
        assert(param);
        start(param->type << ' ' << param->name);
        if (!param->value->isEmpty()) {
          middle(" = " << param->value);
        } else if (param->type->isClass()) {
          middle(" = bi::make_object<" << param->type << ">()");
        }
        finish(';');
      }
      line("");

      /* option flags */
      line("enum {");
      in();
      for (auto param : *o->params) {
        auto name = dynamic_cast<const Parameter*>(param)->name;
        std::string flag = internalise(name->str()) + "FLAG";
        line(flag << ',');
      }
      out();
      line("};");

      /* long options */
      line("int c, option_index;");
      line("option long_options[] = {");
      in();
      for (auto param : *o->params) {
        auto name = dynamic_cast<const Parameter*>(param)->name;
        std::string flag = internalise(name->str()) + "FLAG";

        std::string option = name->str();
        boost::replace_all(option, "_", "-");

        start("{\"");
        middle(option << "\", required_argument, 0, " << flag);
        finish(" },");
      }
      line("{0, 0, 0, 0}");
      out();
      line("};");

      /* short options */
      line("const char* short_options = \"\";");

      /* read in options with getopt_long */
      line("::opterr = 0;");  // handle error reporting ourselves
      start("c = getopt_long_only(argc, argv, short_options, ");
      finish("long_options, &option_index);");
      line("while (c != -1) {");
      in();
      line("switch (c) {");
      in();

      for (auto param : *o->params) {
        auto name = dynamic_cast<const Parameter*>(param)->name;
        std::string flag = internalise(name->str()) + "FLAG";

        line("case " << flag << ':');
        in();
        if (param->type->unwrap()->isBasic()) {
          auto type = dynamic_cast<Named*>(param->type->unwrap());
          assert(type);
          start(name << " = bi::" << type->name);
          finish("(std::string(optarg));");
        } else {
          line(name << " = std::string(optarg);");
        }
        line("break;");
        out();
      }
      out();
      line('}');
      start("c = getopt_long_only(argc, argv, short_options, ");
      finish("long_options, &option_index);");
      out();
      line("}\n");
    }

    /* body of program */
    if (!o->braces->isEmpty()) {
      CppBaseGenerator aux(base, level, header);
      aux << o->braces->strip();
    }

    out();
    line("}\n");
  }
}

void bi::CppBaseGenerator::visit(const BinaryOperator* o) {
  if (!o->braces->isEmpty()) {
    start(o->returnType << ' ');
    if (!header) {
      middle("bi::");
    }
    if (isTranslatable(o->name->str())) {
      middle("operator" << o->name->str());
    } else {
      middle(o->name);
    }
    middle('(');
    middle(o->params->getLeft() << ", " << o->params->getRight());
    middle(')');
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      CppBaseGenerator aux(base, level, false);
      aux << o->braces->strip();
      out();
      finish("}\n");
    }
  }
}

void bi::CppBaseGenerator::visit(const UnaryOperator* o) {
  if (!o->braces->isEmpty()) {
    start(o->returnType << ' ');
    if (!header) {
      middle("bi::");
    }
    if (isTranslatable(o->name->str())) {
      middle("operator" << o->name->str());
    } else {
      middle(o->name);
    }
    middle('(' << o->params << ')');
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      CppBaseGenerator aux(base, level, false);
      aux << o->braces->strip();
      out();
      finish("}\n");
    }
  }
}

void bi::CppBaseGenerator::visit(const AssignmentOperator* o) {
  assert(false);  // should be in CppClassGenerator
}

void bi::CppBaseGenerator::visit(const ConversionOperator* o) {
  assert(false);  // should be in CppClassGenerator
}

void bi::CppBaseGenerator::visit(const Basic* o) {
  // assumed to be built in to compiler library headers
}

void bi::CppBaseGenerator::visit(const Class* o) {
  if (o->isGeneric()) {
    if (header) {
      CppClassGenerator auxClass1(base, level, true);
      CppClassGenerator auxClass2(base, level, false);
      auxClass1 << o;
      auxClass2 << o;
    }
  } else {
    CppClassGenerator auxClass(base, level, header);
    auxClass << o;
  }
}

void bi::CppBaseGenerator::visit(const Alias* o) {
  if (header) {
    line("using " << o->name << " = " << o->base << ';');
  }
}

void bi::CppBaseGenerator::visit(const Generic* o) {
  middle(o->name);
}

void bi::CppBaseGenerator::visit(const Assignment* o) {
  line(o->left << " = " << o->right << ';');
}

void bi::CppBaseGenerator::visit(const ExpressionStatement* o) {
  line(o->single << ';');
}

void bi::CppBaseGenerator::visit(const If* o) {
  line("if " << o->cond << " {");
  in();
  *this << o->braces->strip();
  out();
  if (!o->falseBraces->isEmpty()) {
    line("} else {");
    in();
    *this << o->falseBraces->strip();
    out();
  }
  line("}");
}

void bi::CppBaseGenerator::visit(const For* o) {
  // o->index may be an identifier or a local variable, in the latter case
  // need to ensure that it is only declared once in the first element of the
  // for loop
  auto param = dynamic_cast<LocalVariable*>(o->index);
  if (param) {
    Identifier<LocalVariable> ref(param->name, param->loc, param);
    start("for (" << param << " = " << o->from << "; ");
    middle(&ref << " <= " << o->to << "; ");
    finish("++" << &ref << ") {");
  } else {
    start("for (" << o->index << " = " << o->from << "; ");
    middle(o->index << " <= " << o->to << "; ");
    finish("++" << o->index << ") {");
  }
  in();
  *this << o->braces->strip();
  out();
  line("}");
}

void bi::CppBaseGenerator::visit(const While* o) {
  line("while " << o->cond << " {");
  in();
  *this << o->braces->strip();
  out();
  line("}");
}

void bi::CppBaseGenerator::visit(const DoWhile* o) {
  line("do {");
  in();
  *this << o->braces->strip();
  out();
  line("} while " << o->cond << ';');
}

void bi::CppBaseGenerator::visit(const Assert* o) {
  //if (o->loc) {
  //  line("#line " << o->loc->firstLine << " \"" << o->loc->file->path << '"');
  //}
  line("assert(" << o->cond << ");");
}

void bi::CppBaseGenerator::visit(const Return* o) {
  line("return " << o->single << ';');
}

void bi::CppBaseGenerator::visit(const Yield* o) {
  assert(false);  // should be in CppFiberGenerator
}

void bi::CppBaseGenerator::visit(const Raw* o) {
  if ((header && *o->name == "hpp") || (!header && *o->name == "cpp")) {
    *this << escape_unicode(o->raw);
    if (!std::isspace(o->raw.back())) {
      *this << ' ';
    }
  }
}

void bi::CppBaseGenerator::visit(const StatementList* o) {
  middle(o->head << o->tail);
}

void bi::CppBaseGenerator::visit(const EmptyType* o) {
  middle("void");
}

void bi::CppBaseGenerator::visit(const ArrayType* o) {
  middle("bi::DefaultArray<" << o->single << ',' << o->depth() << '>');
}

void bi::CppBaseGenerator::visit(const TupleType* o) {
  middle("std::tuple<" << o->single << '>');
}

void bi::CppBaseGenerator::visit(const SequenceType* o) {
  middle("bi::Sequence<" << o->single << '>');
}

void bi::CppBaseGenerator::visit(const FunctionType* o) {
  middle("std::function<" << o->returnType << '(' << o->params << ")>");
}

void bi::CppBaseGenerator::visit(const FiberType* o) {
  middle("bi::Fiber<" << o->single << '>');
}

void bi::CppBaseGenerator::visit(const OptionalType* o) {
  middle("boost::optional<" << o->single << '>');
}

void bi::CppBaseGenerator::visit(const ClassType* o) {
  middle("bi::Pointer<bi::type::" << o->name);
  if (!o->typeArgs->isEmpty()) {
    middle('<' << o->typeArgs << '>');
  }
  middle('>');
}

void bi::CppBaseGenerator::visit(const AliasType* o) {
  middle("bi::type::" << o->name);
}

void bi::CppBaseGenerator::visit(const BasicType* o) {
  middle("bi::type::" << o->name);
}

void bi::CppBaseGenerator::visit(const GenericType* o) {
  middle(o->name);
}

void bi::CppBaseGenerator::visit(const TypeIdentifier* o) {
  middle(o->name);
}

void bi::CppBaseGenerator::visit(const TypeList* o) {
  middle(o->head << ", " << o->tail);
}

void bi::CppBaseGenerator::genTemplateParams(const Class* o) {
  if (o->isGeneric()) {
    start("template<");
    for (auto iter = o->typeParams->begin(); iter != o->typeParams->end();
        ++iter) {
      if (iter != o->typeParams->begin()) {
        middle(", ");
      }
      middle("class " << *iter);
    }
    finish('>');
  }
}

void bi::CppBaseGenerator::genTemplateArgs(const Class* o) {
  if (o->isGeneric()) {
    middle('<' << o->typeParams << '>');
  }
}

void bi::CppBaseGenerator::genArgs(const Call* o) {
  middle('(');
  auto iter1 = o->args->begin();
  auto end1 = o->args->end();
  auto iter2 = o->callType->params->begin();
  auto end2 = o->callType->params->end();
  while (iter1 != end1 && iter2 != end2) {
    if (iter1 != o->args->begin()) {
      middle(", ");
    }
    genArg(*iter1, *iter2);
    ++iter1;
    ++iter2;
  }
  middle(')');
}

void bi::CppBaseGenerator::genLeftArg(const BinaryCall* o) {
  genArg(o->args->getLeft(), o->callType->params->getLeft());
}

void bi::CppBaseGenerator::genRightArg(const BinaryCall* o) {
  genArg(o->args->getRight(), o->callType->params->getRight());
}

void bi::CppBaseGenerator::genSingleArg(const UnaryCall* o) {
  genArg(o->args, o->callType->params);
}

void bi::CppBaseGenerator::genArg(const Expression* arg, const Type* type) {
  /* Birch and C++ resolve overloads differently, explicit casting in
   * some situations avoids situations where Birch considers a call
   * unambiguous, whereas C++ does not */
  bool cast = type->isBasic() && !type->equals(*arg->type);
  if (cast) {
    middle("static_cast<" << type->canonical() << ">(");
  }
  middle(arg);
  if (cast) {
    middle(')');
  }
}
