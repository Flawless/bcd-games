#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""DXMT Direct3D 9 shaders: D3D9's "0 * anything = 0" multiply, and a NaN scrub of pixel colour outputs. Both OFF by default.

The Witcher 2 (Steam 20920, 32-bit Direct3D 9, REDengine 2) draws its HUD, minimap and text over a black world at 60 FPS with no
shader or pipeline error in the log. Wine had this bug for the same game (bugs 28883 -> 34266): 0 * inf = NaN in shader arithmetic
reaches the HDR float targets, the eye adaptation carries it from frame to frame, and the tonemapped scene stays black while the UI,
which never passes through it, draws. D3D9 hardware treats 0 * x as 0 for every x; DXVK emulates that with
d3d9.floatEmulation = Strict, its default on drivers with a fast legacy multiply (RADV, NVK, NVIDIA 565.57+), which is how the game runs
on Proton. airconv's DXSO path already caps rcp/rsq/exp/log/pow (DXVK's "Enabled" mode) but lowers mul, mad, lrp, the dot products and
the matrix ops to plain IEEE multiplies.

   MADEIRA_D3D9_FLOAT_STRICT=1  mul, mad, lrp, dp2add/dp3/dp4 and m4x4/m4x3/m3x4/m3x3/m3x2 (all through compute_dot) return 0 in a
       lane where either factor is 0, the way DXVK's Strict mode does (opFMulLegacy / its fma counterpart).
   MADEIRA_D3D9_PS_SANITIZE=1  diagnostic: a NaN in a pixel shader's oC# output is written as 0. If a black scene comes back with this
       alone, NaN reaching the colour targets is proven; FLOAT_STRICT is the fix to keep.

Set in a game's config as env.NAME = 1. The key is read once per process and said once ("[d9-float-strict]", "[d9-ps-sanitize]").
The native shader cache keeps one table per combination: the salt tools/patch-airconv-float-experiments.py adds to
winemetal/unix/cache.c gets these two keys as well, so turning one on never reuses shaders converted without it (and 0/0 keeps the
existing tables).

Native only (airconv + winemetal unix); not in the i386 farm key. Applied after tools/patch-airconv-float-experiments.py (it extends that
patch's cache salt). Idempotent; fails by name if an anchor moves. Run from the repository root.
"""
import pathlib
import sys

ROOT = pathlib.Path("dxmt/src")
MARKER = "madeira-bcd: d9 float strict"


def edit(rel, pairs):
    path = ROOT / rel
    s = path.read_text()
    if MARKER in s:
        print(f"{rel}: already patched")
        return
    for old, new in pairs:
        n = s.count(old)
        if n != 1:
            sys.exit(f"{rel}: anchor found {n} times, expected 1:\n{old[:200]}")
        s = s.replace(old, new)
    path.write_text(s)
    print(f"{rel}: patched")


edit("airconv/dxso_compile.cpp", [
    # --- the two keys, read once per process --------------------------------
    ("""#include "dxso_compile.hpp"
""",
     """#include "dxso_compile.hpp"
/* madeira-bcd: d9 float strict -- MADEIRA_D3D9_FLOAT_STRICT / MADEIRA_D3D9_PS_SANITIZE
 * (tools/patch-airconv-d9-float-strict.py). */
#include <cstdio>
#include <cstdlib>
static bool madeira_d9_key(const char *name, const char *tag) {
  const char *e = std::getenv(name);
  bool on = e && e[0] && e[0] != '0';
  std::fprintf(stderr, "[%s] madeira-bcd %s (%s)\\n", tag, on ? "on" : "off", name);
  return on;
}
static bool madeira_d9_float_strict() {
  static const bool on = madeira_d9_key("MADEIRA_D3D9_FLOAT_STRICT", "d9-float-strict");
  return on;
}
static bool madeira_d9_ps_sanitize() {
  static const bool on = madeira_d9_key("MADEIRA_D3D9_PS_SANITIZE", "d9-ps-sanitize");
  return on;
}
"""),
    # --- the legacy multiply, defined before compute_dot ---------------------
    ("""  auto v4splat = [&](double x) -> Constant * {
    return ConstantVector::getSplat(ElementCount::getFixed(4), ConstantFP::get(Type::getFloatTy(context), x));
  };
""",
     """  auto v4splat = [&](double x) -> Constant * {
    return ConstantVector::getSplat(ElementCount::getFixed(4), ConstantFP::get(Type::getFloatTy(context), x));
  };

  // madeira-bcd: d9 float strict -- D3D9's multiply: 0 * x = 0 for every x
  // (inf and NaN included), lane-wise. DXVK d3d9.floatEmulation = Strict.
  // Plain IEEE multiply with the key off.
  const bool d9_strict = madeira_d9_float_strict();
  auto d9_mul = [&](Value *a, Value *b) -> Value * {
    Value *p = builder.CreateFMul(a, b);
    if (!d9_strict)
      return p;
    Value *zero = Constant::getNullValue(a->getType());
    Value *either_zero = builder.CreateOr(builder.CreateFCmpOEQ(a, zero), builder.CreateFCmpOEQ(b, zero));
    return builder.CreateSelect(either_zero, zero, p);
  };
"""),
    # --- dp2add/dp3/dp4 and the matrix ops -----------------------------------
    ("""    Value *prod = builder.CreateFMul(a, b);
    Value *sum = builder.CreateExtractElement(prod, builder.getInt32(0));""",
     """    Value *prod = d9_mul(a, b);
    Value *sum = builder.CreateExtractElement(prod, builder.getInt32(0));"""),
    # --- mul -----------------------------------------------------------------
    ("""    case DxsoOpcode::Mul:
      fold_binary(ins, [&](Value *a, Value *b) { return builder.CreateFMul(a, b); });""",
     """    case DxsoOpcode::Mul:
      fold_binary(ins, [&](Value *a, Value *b) { return d9_mul(a, b); });"""),
    # --- lrp -----------------------------------------------------------------
    ("""        return builder.CreateFAdd(builder.CreateFMul(s0, builder.CreateFSub(s1, s2)), s2);""",
     """        return builder.CreateFAdd(d9_mul(s0, builder.CreateFSub(s1, s2)), s2);"""),
    # --- mad -----------------------------------------------------------------
    ("""          store_dst(ins.dst, builder.CreateFAdd(builder.CreateFMul(a, b), c));""",
     """          store_dst(ins.dst, builder.CreateFAdd(d9_mul(a, b), c));"""),
    # --- oC# NaN scrub (diagnostic) ------------------------------------------
    ("""        Value *v = builder.CreateLoad(float4Ty, oC_slot[i]);
        if ((unorm_snap >> i) & 1u)""",
     """        Value *v = builder.CreateLoad(float4Ty, oC_slot[i]);
        if (madeira_d9_ps_sanitize()) // madeira-bcd: d9 float strict -- NaN -> 0
          v = builder.CreateSelect(builder.CreateFCmpUNO(v, v), v4splat(0.0), v);
        if ((unorm_snap >> i) & 1u)"""),
])

# --- one native shader-cache table per setting (extends the float experiments' salt) ---
edit("winemetal/unix/cache.c", [
    ("""    salt = clamp * 10 + (b && b[0] == '1');""",
     """    salt = clamp * 10 + (b && b[0] == '1');
    { /* madeira-bcd: d9 float strict -- MADEIRA_D3D9_FLOAT_STRICT / MADEIRA_D3D9_PS_SANITIZE get their own tables */
      const char *fs = getenv("MADEIRA_D3D9_FLOAT_STRICT"), *ps = getenv("MADEIRA_D3D9_PS_SANITIZE");
      salt += 100 * (fs && fs[0] && fs[0] != '0') + 200 * (ps && ps[0] && ps[0] != '0');
    }"""),
])
