import numpy as np
import pyknotid.spacecurves as sp


def _create_knot(arr):
  # note: add_closure does not always work if >0.02 (see src)
  return sp.Knot(arr, add_closure=True, verbose=False)


def number_of_crossings(arr):
  knot = _create_knot(arr)
  return len(knot.raw_crossings(include_closure=False)) // 2


def check_cython():
  try:
    from pyknotid.spacecurves import chelpers

    return True
  except ImportError:
    return False


def gauss_code(arr):
  knot = _create_knot(arr)
  return knot.gauss_code()


def knot_identify(arr):
  knot = _create_knot(arr)
  return knot.identify()


def eq_gauss_code(
  arr1, arr2, allow_flipped_or_mirrored=False, allow_chiral_all=False
):
  gc1 = gauss_code(arr1)
  gc2 = gauss_code(arr2)
  if allow_chiral_all:
    return eq_gauss_code_chiral(gc1, gc2)
  if allow_flipped_or_mirrored:
    return eq_gauss_code_flipped_or_mirrored(gc1, gc2)
  return str(gc1) == str(gc2)


def eq_gauss_code_flipped_or_mirrored(gc1, gc2):
  gc2s = [
    gc2,
    gc2.flipped(),
    gc2.mirrored(),
    gc2.flipped().mirrored(),
    gc2.mirrored().flipped(),
  ]
  return any(str(gc1) == str(gc2) for gc2 in gc2s)


def eq_gauss_code_chiral(gc1, gc2):
  g1 = str(gc1).replace("a", "").replace("c", "")
  g2 = str(gc2).replace("a", "").replace("c", "")
  return g1 == g2


def eq_top(arr1, arr2):
  return knot_identify(arr1) == knot_identify(arr2)


def eq_coord(arr1, arr2):
  # todo: add rotation and translation invariance (subgraph matching)
  return np.allclose(arr1, arr2, atol=1e-3)


def eq_knot(method, arr1, arr2, **kwargs):
  if method == "coord":
    return eq_coord(arr1, arr2)
  if method == "gc":
    return eq_gauss_code(arr1, arr2, **kwargs)
  if method == "top":
    return eq_top(arr1, arr2)
  raise ValueError("Invalid method")


def colorful(x: float) -> np.ndarray:
  """x: float in [0, 1], returns an RGB color in (4,)"""
  if x == 0.0:
    return np.array([1.0, 1.0, 1.0, 1.0])  # white for the start
  r = np.sin(2 * np.pi * (x + 0.0)) * 0.5 + 0.5
  g = np.sin(2 * np.pi * (x + 0.33)) * 0.5 + 0.5
  b = np.sin(2 * np.pi * (x + 0.67)) * 0.5 + 0.5
  return np.array([r, g, b, 1.0])
