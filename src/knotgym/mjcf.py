import logging
import os
import tempfile
import xml.etree.ElementTree as ET
from copy import deepcopy
from itertools import islice

import numpy as np
from etils import epath
import scipy.interpolate as spi


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def split_list(numbers, n, create_overlap=True):
  """
  [1,2,3,4,5], 3 ->
      [[1,2], [3,4], [5]]  # create_overlap=False
      [[1,2,3], [3,4,5], [5,1]]  # create_overlap=True
  """
  # Determine chunk sizes
  k, m = divmod(len(numbers), n)
  it = iter(numbers)
  ret = [list(islice(it, k + (i < m))) for i in range(n)]
  if create_overlap:
    overlaps = [r[0] for r in ret]  # first element of each chunk
    overlaps = overlaps[1:] + [overlaps[0]]  # shift left by 1
    ret = [r + [o] for r, o in zip(ret, overlaps)]
  return ret


def interpolate_3d(coords, N, smoothing_factor=0.001):
  arr = np.array(coords)

  distances = np.cumsum(np.sqrt(np.sum(np.diff(arr, axis=0) ** 2, axis=1)))
  distances = np.insert(distances, 0, 0)  # Include the starting point
  t_new = np.linspace(0, distances[-1], num=N)
  smoothing_factor *= len(arr)  # Adjust this for more or less smoothing

  x_spline = spi.UnivariateSpline(distances, arr[:, 0], s=smoothing_factor)
  y_spline = spi.UnivariateSpline(distances, arr[:, 1], s=smoothing_factor)
  z_spline = spi.UnivariateSpline(distances, arr[:, 2], s=smoothing_factor)

  return np.column_stack([x_spline(t_new), y_spline(t_new), z_spline(t_new)])


def load_xml_from_asset(xml_file: str) -> str:
  if not xml_file.endswith(".xml"):
    xml_file += ".xml"
  return load_from_asset(xml_file)


def load_txt_from_asset(txt_file: str) -> str:
  if not txt_file.endswith(".txt"):
    txt_file += ".txt"
  return load_from_asset(txt_file)


def load_from_asset(file: str) -> str:
  if file == "":
    return file
  if not os.path.exists(file):
    file = epath.resource_path("knotgym") / "assets" / file
  return str(file)


def initialize_knot_coords(
  template_xml_file: str = "unknot.xml",
  data_file: str = "initial.txt",
  num_subcables: int = 7,
  num_beads: int = 100,
) -> str:
  """Dynamically create mjcf file from template file given configurations.

  Args:
      template_xml_file (str): template mjcf file name.
      data_file (str): data file name containing nx3 vertices.
      num_subcables (int, optional): number of subcables.
          This improves mj_step at the cost of numerical stability.
          Defaults to 1.
      num_beads (int, optional): number of beads.
          If num_beads != n, then data_file will be interpolated.
          The more beads, the more granular movement, the slower.
          Defaults to 100.

  Returns:
      (str): temporary mjcf file name.
  """
  template_xml_file = load_xml_from_asset(template_xml_file)
  data_file = load_txt_from_asset(data_file)
  if not os.path.exists(data_file):
    logger.warning(
      f"{data_file} does not exist, returning {template_xml_file}."
    )
    return template_xml_file
  logger.info(f"building mjcf from {template_xml_file}")
  logger.info(f"initializing vertices with {data_file}")
  tree = ET.parse(template_xml_file)
  root = tree.getroot()
  worldbody = root.find(".//worldbody")

  vertices = np.loadtxt(data_file)  # n x 3

  # prepare vertices, interpolate if necessary
  if num_beads != len(vertices):
    logger.info(f"interpolate ({len(vertices)}) to num_beads({num_beads})")
    vertices = np.vstack([vertices, vertices[0]])
    new_vertices = interpolate_3d(vertices, num_beads + 1)
    vertices = new_vertices[:-1]  # remove the last one

  vertices = [" ".join(map(str, r)) for r in vertices]  # list of strings
  verticess = split_list(vertices, num_subcables)
  verticess = [" ".join(v) for v in verticess]
  prefixes = [f"Subc_{i}_" for i in range(num_subcables)]

  # rewrite composites
  composites = root.findall(".//composite[@type='cable']")
  if len(composites) == 1:
    assert len(composites) == 1, "Expect one template cable object"
    composite = composites[0]
    composite.attrib.pop("count")
    composite.attrib.pop("curve")
    composite.attrib.pop("size")
    composite.attrib.pop("offset")
    composite_template = deepcopy(composite)
    worldbody.remove(composite)
    del composite, composites
  else:
    composite_template = deepcopy(composites[0])
    # remove all composites from root
    for composite in composites:
      worldbody.remove(composite)
    del composites
  composites = []
  for vertices, prefix in zip(verticess, prefixes):
    composite = deepcopy(composite_template)
    composite.set("vertex", vertices)
    composite.set("prefix", prefix)
    composites.append(composite)
  worldbody.extend(composites)

  # rewrite equality
  equality = root.find(".//equality")
  eq_template = deepcopy(equality[0])
  equality.clear()
  eqs = []
  for i in range(num_subcables):
    eq = deepcopy(eq_template)
    eq.set("site1", f"Subc_{i}_S_last")
    eq.set("site2", f"Subc_{(i + 1) % num_subcables}_S_first")
    eqs.append(eq)
  equality.extend(eqs)

  # rewrite contact
  contact = root.find(".//contact")
  contact.clear()
  contact.extend(
    [
      ET.Element(
        "exclude",
        attrib={
          "body1": f"Subc_{i}_B_last",
          "body2": f"Subc_{(i + 1) % num_subcables}_B_first",
        },
      )
      for i in range(num_subcables)
    ]
  )

  # save to temporary file
  with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
    ET.indent(tree)
    tree.write(f, encoding="unicode")
  logger.info(f"temporary mjcf saved in {f.name}")
  return f.name
