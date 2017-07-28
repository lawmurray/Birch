/**
 * @file
 */
#include "bi/io/cpp/CppBaseGenerator.hpp"

#include "bi/io/cpp/CppClassGenerator.hpp"
#include "bi/io/cpp/CppCoroutineGenerator.hpp"
#include "bi/visitor/Gatherer.hpp"
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

void bi::CppBaseGenerator::visit(const List<Expression>* o) {
  middle(o->head);
  if (o->tail) {
    middle(", " << o->tail);
  }
}

void bi::CppBaseGenerator::visit(const Literal<bool>* o) {
  middle(o->str);
}

void bi::CppBaseGenerator::visit(const Literal<int64_t>* o) {
  middle(o->str);
}

void bi::CppBaseGenerator::visit(const Literal<double>* o) {
  middle(o->str);
}

void bi::CppBaseGenerator::visit(const Literal<const char*>* o) {
  middle("std::string(" << o->str << ')');
}

void bi::CppBaseGenerator::visit(const Parentheses* o) {
  middle('(' << o->single << ')');
}

void bi::CppBaseGenerator::visit(const Brackets* o) {
  middle("make_view(" << o->single << ')');
}

void bi::CppBaseGenerator::visit(const Call* o) {
  middle(o->single << o->parens);
}

void bi::CppBaseGenerator::visit(const OverloadedCall<Function>* o) {
  middle(o->single << o->parens);
}

void bi::CppBaseGenerator::visit(const OverloadedCall<Coroutine>* o) {
  middle(o->single << o->parens);
}

void bi::CppBaseGenerator::visit(const OverloadedCall<MemberFunction>* o) {
  middle(o->single << o->parens);
}

void bi::CppBaseGenerator::visit(const OverloadedCall<MemberCoroutine>* o) {
  middle(o->single << o->parens);
}

void bi::CppBaseGenerator::visit(const OverloadedCall<BinaryOperator>* o) {
  if (isTranslatable(o->name->str())) {
    /* can use as raw C++ operator */
    genArg(o->left.get(), o->target->left.get());
    middle(' ' << o->name->str() << ' ');
    genArg(o->right.get(), o->target->right.get());
  } else {
    /* must use as function */
    middle("bi::" << o->name << '(');
    genArg(o->left.get(), o->target->left.get());
    middle(", ");
    genArg(o->right.get(), o->target->right.get());
    middle(')');
  }
}

void bi::CppBaseGenerator::visit(const OverloadedCall<UnaryOperator>* o) {
  if (isTranslatable(o->name->str())) {
    /* can use as raw C++ operator */
    middle(o->name->str());
    genArg(o->single.get(), o->target->single.get());
  } else {
    /* must use as function */
    middle("bi::" << o->name << '(');
    genArg(o->single.get(), o->target->single.get());
    middle(')');
  }
}

void bi::CppBaseGenerator::visit(const Slice* o) {
  middle(o->single << '(' << o->brackets << ')');
}

void bi::CppBaseGenerator::visit(const LambdaFunction* o) {
  middle("[&]" << o->parens << " {");
  in();
  *this << o->braces;
  out();
  start("}");
}

void bi::CppBaseGenerator::visit(const Span* o) {
  middle("make_span(" << o->single << ')');
}

void bi::CppBaseGenerator::visit(const Index* o) {
  middle("make_index(" << o->single << " - 1)");
}

void bi::CppBaseGenerator::visit(const Range* o) {
  middle("make_range(" << o->left << " - 1, " << o->right << " - 1)");
}

void bi::CppBaseGenerator::visit(const Member* o) {
  const This* leftThis = dynamic_cast<const This*>(o->left.get());
  const Super* leftSuper = dynamic_cast<const Super*>(o->left.get());
  if (leftThis) {
    // tidier this way
    middle("this->");
  } else if (leftSuper) {
    // tidier this way
    middle("super_type::");
  } else {
    middle(o->left);
    if (o->left->type->isClass()) {
      middle("->");
    } else {
      middle('.');
    }
  }
  middle(o->right);
}

void bi::CppBaseGenerator::visit(const Super* o) {
  // only need to handle the case outside member expression, which shouldn't
  // exist
  assert(false);
}

void bi::CppBaseGenerator::visit(const This* o) {
  // only need to handle the case outside member expression, where must
  // ensure correct global or fiber-local pointer is used
  middle("pointer_from_this<this_type>()");
}

void bi::CppBaseGenerator::visit(const Parameter* o) {
  if (!o->type->assignable) {
    middle("const ");
  }
  middle(o->type << "& " << o->name);
  if (!o->value->isEmpty()) {
    middle(" = " << o->value);
  }
}

void bi::CppBaseGenerator::visit(const MemberParameter* o) {
  if (!o->type->assignable) {
    middle("const ");
  }
  middle(o->type << "& " << o->name);
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
  middle("bi::func::" << o->name);
}

void bi::CppBaseGenerator::visit(const OverloadedIdentifier<Coroutine>* o) {
  middle("bi::func::" << o->name);
}

void bi::CppBaseGenerator::visit(const OverloadedIdentifier<MemberFunction>* o) {
  middle(o->name);
}

void bi::CppBaseGenerator::visit(const OverloadedIdentifier<MemberCoroutine>* o) {
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

    /* standard headers */
    line("#include \"bi/libbirch.hpp\"");
  } else {
    /* include main header file */
    boost::filesystem::path file(o->path);
    file.replace_extension(".hpp");
    line("#include \"" << file.filename().string() << "\"\n");

    line("");
  }

  /* main code */
  *this << o->root;
}

void bi::CppBaseGenerator::visit(const Import* o) {
  if (header) {
    boost::filesystem::path file = o->path->file();
    file.replace_extension(".hpp");
    line("#include \"" << file.string() << "\"");
  }
}

void bi::CppBaseGenerator::visit(const GlobalVariable* o) {
  if (header) {
    line("namespace bi {");
    line("extern " << o->type << ' ' << o->name << ';');
    line("}\n");
  } else {
    start(o->type << " bi::" << o->name);
    genInit(o);
    finish(';');
  }
}

void bi::CppBaseGenerator::visit(const LocalVariable* o) {
  start(o->type << ' ' << o->name);
  genInit(o);
  finish(';');
}

void bi::CppBaseGenerator::visit(const MemberVariable* o) {
  assert(false);  // should be in CppClassGenerator
}

void bi::CppBaseGenerator::visit(const Function* o) {
  if (!o->braces->isEmpty()) {
    if (header) {
      line("namespace bi {");
      in();
      line("namespace func {");
      out();
    }

    start(o->returnType << ' ');
    if (!header) {
      middle("bi::func::");
    }
    middle(o->name << o->parens);

    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();

      /* body */
      CppBaseGenerator aux(base, level, false);
      aux << o->braces;

      out();
      finish("}\n");
    }
    if (header) {
      in();
      line("}");
      out();
      line("}\n");
    }
  }
}

void bi::CppBaseGenerator::visit(const Coroutine* o) {
  CppCoroutineGenerator auxCoroutine(base, level, header);
  auxCoroutine << o;
}

void bi::CppBaseGenerator::visit(const MemberFunction* o) {
  assert(false);  // should be in CppClassGenerator
}

void bi::CppBaseGenerator::visit(const MemberCoroutine* o) {
  assert(false);  // should be in CppClassGenerator
}

void bi::CppBaseGenerator::visit(const Program* o) {
  if (header) {
    line("namespace bi {");
    in();
    line("namespace program {");
    out();
    line("extern \"C\" void " << o->name << "(int argc, char** argv);");
    in();
    line("}");
    out();
    line("}\n");
  } else {
    line("void bi::program::" << o->name << "(int argc, char** argv) {");
    in();
    if (o->parens->tupleSize() > 0) {
      /* option variables */
      for (auto iter = o->parens->begin(); iter != o->parens->end(); ++iter) {
        auto param = dynamic_cast<const Parameter*>(*iter);
        assert(param);
        start(param->type << ' ' << param->name);
        if (!param->value->isEmpty()) {
          middle(" = " << param->value);
        } else if (param->type->isClass()) {
          auto type = dynamic_cast<const ClassType*>(param->type->strip());
          assert(type);
          middle(" = make_object<bi::type::" << type->name << ">()");
        }
        finish(';');
      }
      line("");

      /* option flags */
      line("enum {");
      in();
      for (auto iter = o->parens->begin(); iter != o->parens->end(); ++iter) {
        std::string flag = dynamic_cast<const Parameter*>(*iter)->name->str()
            + "_ARG";
        boost::to_upper(flag);
        start(flag);
        if (iter == o->parens->begin()) {
          middle(" = 256");
        }
        finish(',');
      }
      out();
      line("};");

      /* long options */
      line("int c, option_index;");
      line("option long_options[] = {");
      in();
      for (auto iter = o->parens->begin(); iter != o->parens->end(); ++iter) {
        const std::string& name =
            dynamic_cast<const Parameter*>(*iter)->name->str();
        //if (name.length() > 1) {
        std::string flag = name + "_ARG";
        boost::to_upper(flag);
        std::string option = name;
        boost::replace_all(option, "_", "-");

        line(
            "{\"" << option << "\", required_argument, 0, " << flag << " },");
        //}
      }
      line("{0, 0, 0, 0}");
      out();
      line("};");

      /* short options */
      start("const char* short_options = \"");
      //for (auto iter = o->parens->begin(); iter != o->parens->end(); ++iter) {
      //  const std::string& name = dynamic_cast<const Parameter*>(*iter)->name->str();
      //  if (name.length() == 1) {
      //    middle(name << ':');
      //  }
      //}
      finish("\";");

      /* read in options with getopt_long */
      line("::opterr = 0; // handle error reporting ourselves");
      line(
          "c = getopt_long_only(argc, argv, short_options, long_options, &option_index);");
      line("while (c != -1) {");
      in();
      line("switch (c) {");
      in();

      for (auto iter = o->parens->begin(); iter != o->parens->end(); ++iter) {
        auto name = dynamic_cast<const Named*>(*iter)->name;
        assert(name);
        std::string flag = name->str() + "_ARG";
        boost::to_upper(flag);

        start("case ");
        //if (name.length() > 1) {
        middle(flag);
        //} else {
        //  middle('\'' << name << '\'');
        //}
        finish(':');
        in();
        if ((*iter)->type->isBasic()) {
          auto type = dynamic_cast<Named*>((*iter)->type.get());
          assert(type);
          line(name << " = bi::func::" << type->name << "(optarg);");
        } else if ((*iter)->type->isClass()) {
          line('*' << name << " = optarg;");
        } else {
          line(name << " = optarg;");
        }
        line("break;");
        out();
      }
      //line("default:");
      //in();
      //line("throw UnknownOptionException(argv[optind - 1]);");
      //line("break;");
      //out();
      out();
      line('}');
      line(
          "c = getopt_long_only(argc, argv, short_options, long_options, &option_index);");
      out();
      line("}\n");
    }

    /* body of program */
    if (!o->braces->isEmpty()) {
      CppBaseGenerator aux(base, level, header);
      aux << o->braces;
    }

    out();
    line("}\n");
  }
}

void bi::CppBaseGenerator::visit(const BinaryOperator* o) {
  if (!o->braces->isEmpty()) {
    if (header) {
      line("namespace bi {");
    }

    start(o->returnType << ' ');
    if (!header) {
      middle("bi::");
    }
    if (isTranslatable(o->name->str())) {
      middle("operator" << o->name->str());
    } else {
      middle(o->name);
    }
    middle('(' << o->left << ", " << o->right << ')');
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      CppBaseGenerator aux(base, level, false);
      aux << o->braces;
      out();
      finish("}\n");
    }
    if (header) {
      line("}\n");
    }
  }
}

void bi::CppBaseGenerator::visit(const UnaryOperator* o) {
  if (!o->braces->isEmpty()) {
    if (header) {
      line("namespace bi {");
    }
    start(o->returnType << ' ');
    if (!header) {
      middle("bi::");
    }
    if (isTranslatable(o->name->str())) {
      middle("operator" << o->name->str());
    } else {
      middle(o->name);
    }
    middle('(' << o->single << ')');
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      CppBaseGenerator aux(base, level, false);
      aux << o->braces;
      out();
      finish("}\n");
    }
    if (header) {
      line("}\n");
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
  if (header) {
    line("namespace bi {");
    in();
    line("namespace type {");
    out();
  }
  CppClassGenerator auxClass(base, level, header);
  auxClass << o;
  if (header) {
    in();
    line("}");
    out();
    line("}\n");
  }
}

void bi::CppBaseGenerator::visit(const Alias* o) {
  if (header) {
    line("namespace bi {");
    in();
    line("namespace type {");
    out();
    line("using " << o->name << " = " << o->base << ';');
    in();
    line("}");
    out();
    line("}\n");
  }
}

void bi::CppBaseGenerator::visit(const List<Statement>* o) {
  middle(o->head);
  if (o->tail) {
    middle(o->tail);
  }
}

void bi::CppBaseGenerator::visit(const Assignment* o) {
  if (*o->name == "<~" || *o->name == "~") {
    start(o->name << '(' << o->left << ", " << o->right << ')');
  } else {
    if (o->left->type->isClass() && !o->right->type->isClass()) {
      start("*(" << o->left << ')');
    } else {
      start(o->left);
    }
    middle(" = ");
    if (!o->left->type->isClass() && o->right->type->isClass()) {
      middle("*(" << o->right << ')');
    } else {
      middle(o->right);
    }
  }
  finish(';');
}

void bi::CppBaseGenerator::visit(const ExpressionStatement* o) {
  line(o->single << ';');
}

void bi::CppBaseGenerator::visit(const If* o) {
  line("if " << o->cond << " {");
  in();
  *this << o->braces;
  out();
  if (!o->falseBraces->isEmpty()) {
    line("} else {");
    in();
    *this << o->falseBraces;
    out();
  }
  line("}");
}

void bi::CppBaseGenerator::visit(const For* o) {
  ///@todo May need to be more sophisticated to accommodate arbitrary types
  line(
      "for (" << o->index << " = " << o->from << "; " << o->index << " <= " << o->to << "; ++" << o->index << ") {");
  in();
  *this << o->braces;
  out();
  line("}");
}

void bi::CppBaseGenerator::visit(const While* o) {
  line("while " << o->cond << " {");
  in();
  *this << o->braces;
  out();
  line("}");
}

void bi::CppBaseGenerator::visit(const Assert* o) {
  line("assert(" << o->cond << ");");
}

void bi::CppBaseGenerator::visit(const Return* o) {
  line("return " << o->single << ';');
}

void bi::CppBaseGenerator::visit(const Yield* o) {
  assert(false);  // should be in CppCoroutineGenerator
}

void bi::CppBaseGenerator::visit(const Raw* o) {
  if ((header && *o->name == "hpp") || (!header && *o->name == "cpp")) {
    *this << o->raw;
    if (!std::isspace(o->raw.back())) {
      *this << ' ';
    }
  }
}

void bi::CppBaseGenerator::visit(const EmptyType* o) {
  middle("void");
}

void bi::CppBaseGenerator::visit(const List<Type>* o) {
  middle(o->head);
  Type* tail = o->tail.get();
  List<Type>* list = dynamic_cast<List<Type>*>(tail);
  while (list) {
    middle(',' << list->head);
    tail = list->tail.get();
    list = dynamic_cast<List<Type>*>(tail);
  }
  middle(',' << tail);
}

void bi::CppBaseGenerator::visit(const ArrayType* o) {
  middle("bi::DefaultArray<" << o->single << ',' << o->count() << '>');
}

void bi::CppBaseGenerator::visit(const ParenthesesType* o) {
  middle('(' << o->single << ')');
}

void bi::CppBaseGenerator::visit(const FunctionType* o) {
  middle("std::function<" << o->returnType << o->parens);
}

void bi::CppBaseGenerator::visit(const FiberType* o) {
  middle("bi::Fiber<" << o->returnType << ">");
}

void bi::CppBaseGenerator::visit(const ClassType* o) {
  middle("bi::Pointer<bi::type::" << o->name << ">");
}

void bi::CppBaseGenerator::visit(const AliasType* o) {
  middle(o->target->base);
}

void bi::CppBaseGenerator::visit(const BasicType* o) {
  middle("bi::type::" << o->name);
}

void bi::CppBaseGenerator::genArgs(const Expression* args,
    const Expression* params) {
  auto iter1 = args->begin();
  auto iter2 = params->begin();
  while (iter1 != args->end() && iter2 != params->end()) {
    if (iter1 != args->begin()) {
      middle(", ");
    }
    genArg(*iter1, *iter2);
    ++iter1;
    ++iter2;
  }
  assert(iter1 == args->end());
  assert(iter2 == params->end());
}

void bi::CppBaseGenerator::genArg(const Expression* arg,
    const Expression* param) {
  if (arg->type->isClass() && !param->type->isClass()) {
    middle("*(" << arg << ')');
  } else {
    middle(arg);
  }
}
