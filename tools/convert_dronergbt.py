#!/usr/bin/env python3
"""
Strict DroneRGBT converter: directly parse XML `<object>/<point>/<x,y>` and copy
corresponding RGB and Infrared images into flat `out/train` and `out/test` folders.

This script avoids any heuristics and is straightforward for the DroneRGBT layout:
  src/Train/RGB/<base>.jpg
  src/Train/Infrared/<base>R.jpg
  src/Train/GT_/<base>R.xml

Writes:
  out/train/<base>_RGB.jpg
  out/train/<base>_T.jpg
  out/train/<base>_GT.npy

And same for `Test` -> `out/test`.
"""

import argparse
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import numpy as np


def parse_points_from_xml(xml_path: Path):
    try:
        root = ET.parse(str(xml_path)).getroot()
    except Exception:
        return np.empty((0, 2), dtype=np.float32)
    pts = []
    for ptag in root.findall('.//object/point'):
        # prefer findtext which returns None or the text
        x_text = ptag.findtext('x') or ptag.findtext('X')
        y_text = ptag.findtext('y') or ptag.findtext('Y')
        if not x_text or not y_text:
            continue
        try:
            xf = float(x_text.strip())
            yf = float(y_text.strip())
        except Exception:
            continue
        pts.append((xf, yf))
    if not pts:
        return np.empty((0, 2), dtype=np.float32)
    return np.array(pts, dtype=np.float32)


def convert_split(src_root: Path, split: str, out_root: Path):
    split_dir = src_root / split
    if not split_dir.exists():
        return 0
    gt_dir = split_dir / 'GT_'
    xml_files = list(gt_dir.glob('*.xml')) if gt_dir.exists() else list(split_dir.rglob('*.xml'))
    out_dir = out_root / split.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for xml in xml_files:
        stem = xml.stem
        # expect trailing R
        if stem.lower().endswith('_r'):
            base = stem[:-2]
        elif stem.lower().endswith('r'):
            base = stem[:-1]
        else:
            continue
        rgb = split_dir / 'RGB' / (base + '.jpg')
        ir = split_dir / 'Infrared' / (base + 'R.jpg')
        if not rgb.exists() or not ir.exists():
            # try other extensions
            found_rgb = None
            found_ir = None
            for p in split_dir.rglob(base + '.*'):
                if p.suffix.lower() in ['.jpg', '.png', '.jpeg'] and 'rgb' in p.parent.name.lower():
                    found_rgb = p
            for p in split_dir.rglob(base + 'R.*'):
                if p.suffix.lower() in ['.jpg', '.png', '.jpeg'] and 'infra' in p.parent.name.lower():
                    found_ir = p
            if found_rgb:
                rgb = found_rgb
            if found_ir:
                ir = found_ir
        if not rgb.exists() or not ir.exists():
            continue

        pts = parse_points_from_xml(xml)
        # write files
        shutil.copy2(str(rgb), str(out_dir / (base + '_RGB.jpg')))
        shutil.copy2(str(ir), str(out_dir / (base + '_T.jpg')))
        np.save(str(out_dir / (base + '_GT.npy')), pts.astype(np.float32))
        count += 1
    return count


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    src = Path(args.src)
    out = Path(args.out)
    if not src.exists():
        print('src missing')
        return
    total = 0
    for s in ['Train', 'Test']:
        c = convert_split(src, s, out)
        print('converted', c, 'from', s)
        total += c
    print('total', total)


if __name__ == '__main__':
    main()
