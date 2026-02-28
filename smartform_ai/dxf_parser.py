"""
dxf_parser.py
Extracts structural elements from AutoCAD DXF files (.dxf) using ezdxf.

Strategy:
  1.  Read all layers present in the file.
  2.  User tells us which layer → Column / Slab / Beam.
  3.  For each entity on a mapped layer:
        - Closed LWPOLYLINE / POLYLINE  → bounding-box gives L × W (footprint)
        - LINE / open LWPOLYLINE        → length becomes beam/span length
        - INSERT (block reference)      → pull ATTRIB tags for dimensions
  4.  User provides default Height / Width / Casting Date / Cost per type.
  5.  Returns a SmartForm-ready DataFrame.
"""

import math
import re
import pandas as pd
from datetime import datetime

try:
    import ezdxf
    EZDXF_OK = True
except ImportError:
    EZDXF_OK = False


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _bbox(points):
    """Return (min_x, min_y, max_x, max_y) of a list of (x, y[, z, ...]) tuples."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)

def _lwpoly_dims(entity):
    """Return (length_m, width_m) from a closed LWPOLYLINE bounding box."""
    pts = list(entity.get_points('xy'))
    if len(pts) < 3:
        return None, None
    x0, y0, x1, y1 = _bbox(pts)
    l = round(abs(x1 - x0) / 1000, 4)   # DXF units assumed mm
    w = round(abs(y1 - y0) / 1000, 4)
    return (max(l, w), min(l, w))         # ensure L >= W

def _line_length(entity):
    """Return length of a LINE entity in metres (assuming mm units)."""
    s = entity.dxf.start
    e = entity.dxf.end
    d = math.sqrt((e.x - s.x)**2 + (e.y - s.y)**2 + (e.z - s.z)**2)
    return round(d / 1000, 4)

def _attrib_dict(insert_entity):
    """Collect ATTRIB tag→value pairs from a block INSERT."""
    attrs = {}
    try:
        for attrib in insert_entity.attribs:
            attrs[attrib.dxf.tag.strip().lower()] = attrib.dxf.text.strip()
    except Exception:
        pass
    return attrs


# ── Public API ────────────────────────────────────────────────────────────────

def get_layers(file_obj) -> list[str]:
    """
    Read a DXF file and return all layer names present in model space.
    file_obj : file-like object (Streamlit UploadedFile) or str path.
    """
    if not EZDXF_OK:
        return []
    try:
        if hasattr(file_obj, 'read'):
            import tempfile, os
            suffix = '.dxf'
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(file_obj.read())
                tmp_path = tmp.name
            doc = ezdxf.readfile(tmp_path)
            os.unlink(tmp_path)
        else:
            doc = ezdxf.readfile(file_obj)

        msp    = doc.modelspace()
        layers = sorted({e.dxf.layer for e in msp if hasattr(e, 'dxf') and hasattr(e.dxf, 'layer')})
        return layers
    except Exception as err:
        return [f"ERROR: {err}"]


def parse_dxf(
    file_obj,
    layer_map: dict,          # {'Layer-Name': 'Column'|'Slab'|'Beam', ...}
    default_height: dict = None,   # {'Column': 3.0, 'Slab': 0.2, 'Beam': 0.5}
    default_width:  dict = None,   # {'Beam': 0.3}
    default_cost:   dict = None,   # {'Column': 1200, 'Slab': 3500, 'Beam': 1800}
    default_date:   str  = None,
    default_zone:   str  = 'Zone-A',
) -> pd.DataFrame:
    """
    Parse a DXF file and return a SmartForm-ready DataFrame.

    Parameters
    ----------
    file_obj    : Streamlit UploadedFile or file path string
    layer_map   : Maps DXF layer names to element types
                  e.g. {'S-COLS': 'Column', 'S-SLAB': 'Slab'}
    default_height : Fallback height per element type (metres)
    default_width  : Fallback width for Beam elements
    default_cost   : Cost per set per element type
    default_date   : Casting date fallback (YYYY-MM-DD)
    default_zone   : Zone label
    """
    if not EZDXF_OK:
        raise ImportError("ezdxf is not installed. Run: pip install ezdxf")

    if default_height is None:
        default_height = {'Column': 3.0, 'Slab': 0.2, 'Beam': 0.5}
    if default_width is None:
        default_width  = {'Column': None, 'Slab': None, 'Beam': 0.3}
    if default_cost is None:
        default_cost   = {'Column': 1200, 'Slab': 3500, 'Beam': 1800}
    if default_date is None:
        default_date   = datetime.today().strftime('%Y-%m-%d')

    # Write to temp file for ezdxf
    import tempfile, os as _os
    if hasattr(file_obj, 'read'):
        raw = file_obj.read()
        with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
    else:
        tmp_path = file_obj

    try:
        doc = ezdxf.readfile(tmp_path)
    finally:
        if hasattr(file_obj, 'read'):
            _os.unlink(tmp_path)

    msp   = doc.modelspace()
    rows  = []
    count = 0

    for entity in msp:
        layer = getattr(entity.dxf, 'layer', None)
        if layer not in layer_map:
            continue

        elem_type = layer_map[layer]
        etype     = entity.dxftype()
        h = default_height.get(elem_type, 3.0)
        cost = default_cost.get(elem_type, 1500)
        count += 1

        length, width = None, None

        # ── Closed polyline → column/slab footprint ───────────────────────────
        if etype in ('LWPOLYLINE', 'POLYLINE'):
            try:
                is_closed = entity.is_closed if hasattr(entity, 'is_closed') else entity.dxf.flags & 1
            except Exception:
                is_closed = False

            if etype == 'LWPOLYLINE':
                length, width = _lwpoly_dims(entity)
            else:
                pts = [v.dxf.location for v in entity.vertices]
                if pts:
                    x0, y0, x1, y1 = _bbox([(p.x, p.y) for p in pts])
                    length = round(abs(x1 - x0) / 1000, 4)
                    width  = round(abs(y1 - y0) / 1000, 4)
                    if length < width:
                        length, width = width, length

        # ── Line → beam/span length ───────────────────────────────────────────
        elif etype == 'LINE':
            length = _line_length(entity)
            width  = default_width.get(elem_type, 0.3)

        # ── Block INSERT → try ATTRIB tags for dimensions ────────────────────
        elif etype == 'INSERT':
            attrs = _attrib_dict(entity)

            def _attr_metres(keys):
                for k in keys:
                    if k in attrs:
                        try:
                            v = float(re.sub(r'[^\d.]', '', attrs[k]))
                            return round(v / 1000, 4)
                        except Exception:
                            pass
                return None

            length = _attr_metres(['length', 'len', 'l', 'width', 'b'])
            width  = _attr_metres(['width', 'b', 'w', 'depth', 'd'])
            h_attr = _attr_metres(['height', 'h', 'depth', 'thk'])
            if h_attr:
                h = h_attr

        if length is None or length == 0:
            continue
        if width is None or width == 0:
            width = default_width.get(elem_type) or length  # square fallback

        rows.append({
            'Element_ID':             f"DXF-{count:04d}",
            'Type':                   elem_type,
            'Length':                 length,
            'Width':                  width,
            'Height':                 h,
            'Floor':                  1,
            'Zone':                   default_zone,
            'Casting_Date':           default_date,
            'Formwork_Cost_per_Set':  cost,
            'Replacement_Cost_per_Set': round(cost * 0.85, 0),
            'Max_Reuse_Count':        10,
            '_source':                f"DXF:{layer}",
        })

    return pd.DataFrame(rows)
