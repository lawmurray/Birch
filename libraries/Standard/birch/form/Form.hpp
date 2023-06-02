/**
 * @file
 */
#pragma once

/*
 * Add `o.` to the start of each argument, e.g. `BIRCH_O_DOT(a, b, c)` yields
 * `o.a, o.b, o.c`.
 */
#define BIRCH_O_DOT3(arg) o.arg)
#define BIRCH_O_DOT2(arg, ...) o.arg __VA_OPT__(, BIRCH_O_DOT3(__VA_ARGS__))
#define BIRCH_O_DOT1(arg, ...) o.arg __VA_OPT__(, BIRCH_O_DOT2(__VA_ARGS__))
#define BIRCH_O_DOT0(arg, ...) o.arg __VA_OPT__(, BIRCH_O_DOT1(__VA_ARGS__))
#define BIRCH_O_DOT(arg, ...) o.arg __VA_OPT__(, BIRCH_O_DOT0(__VA_ARGS__))
