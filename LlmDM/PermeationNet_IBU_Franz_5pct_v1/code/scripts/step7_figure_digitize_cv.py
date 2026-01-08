import json
import math
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import cv2

IN_TRIAGE = Path("GenAI/outputs/figure_triage_v1.csv")
OUT_CURVES = Path("GenAI/outputs/figure_digitized_curves.jsonl")
OUT_ENDPTS = Path("GenAI/outputs/figure_digitized_endpoints.csv")

MIN_EDGE_POINTS = 800
MIN_CLUSTER_POINTS = 250

def parse_bbox(v) -> Optional[List[float]]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return None
    return None

def crop_by_bbox(img: np.ndarray, bbox: List[float]) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = bbox
    x0 = int(max(0, min(w - 2, round(float(x0) * w))))
    x1 = int(max(x0 + 1, min(w - 1, round(float(x1) * w))))
    y0 = int(max(0, min(h - 2, round(float(y0) * h))))
    y1 = int(max(y0 + 1, min(h - 1, round(float(y1) * h))))
    return img[y0:y1, x0:x1].copy()

def remove_axes_edges(edges: np.ndarray) -> np.ndarray:
    """Erase long straight lines (axes) using HoughLinesP."""
    h, w = edges.shape[:2]
    min_len = int(0.65 * min(h, w))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=120,
                            minLineLength=min_len, maxLineGap=10)
    out = edges.copy()
    if lines is not None:
        for l in lines[:, 0, :]:
            x1, y1, x2, y2 = l
            cv2.line(out, (x1, y1), (x2, y2), 0, thickness=7)
    return out

def extract_edge_mask(crop_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = remove_axes_edges(edges)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(edges, kernel, iterations=1)
    return mask

def kmeans_cluster_colors(hsv_pixels: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    hsv_pixels: Nx3 float32
    return labels (N,), centers (k,3)
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    attempts = 3
    compactness, labels, centers = cv2.kmeans(
        hsv_pixels.astype(np.float32), k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS
    )
    return labels.flatten(), centers

def build_curve_from_points(xs: np.ndarray, ys: np.ndarray, min_cols: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """Per x column, pick median y. Return sorted pixel coords."""
    if xs.size < 50:
        return np.array([]), np.array([])
    pts: Dict[int, List[int]] = {}
    for x, y in zip(xs, ys):
        pts.setdefault(int(x), []).append(int(y))
    x_cols = sorted(pts.keys())
    if len(x_cols) < min_cols:
        return np.array([]), np.array([])
    x_out, y_out = [], []
    for x in x_cols:
        y_out.append(int(np.median(pts[x])))
        x_out.append(x)
    return np.array(x_out, dtype=np.float32), np.array(y_out, dtype=np.float32)

def pixel_to_data(xpix: np.ndarray, ypix: np.ndarray,
                  x_min: float, x_max: float,
                  y_min: float, y_max: float,
                  w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    x = x_min + (xpix / max(1.0, (w - 1))) * (x_max - x_min)
    y = y_max - (ypix / max(1.0, (h - 1))) * (y_max - y_min)
    return x, y

def interp_at(x: np.ndarray, y: np.ndarray, xq: float) -> float:
    if x.size < 2:
        return float("nan")
    idx = np.argsort(x)
    xs = x[idx]
    ys = y[idx]
    # clip query into range
    xq2 = min(max(xq, float(xs[0])), float(xs[-1]))
    return float(np.interp(xq2, xs, ys))

def coarse_color_name(hsv_center: np.ndarray) -> str:
    h, s, v = hsv_center
    # OpenCV HSV: H in [0,179]
    if v < 40:
        return "black"
    if s < 40 and v > 180:
        return "gray"
    if s < 40:
        return "gray"
    hue = float(h) * 2.0  # -> degrees
    if hue < 15 or hue >= 345:
        return "red"
    if hue < 45:
        return "orange"
    if hue < 75:
        return "yellow"
    if hue < 150:
        return "green"
    if hue < 210:
        return "cyan"
    if hue < 270:
        return "blue"
    if hue < 330:
        return "purple"
    return "red"

def main():
    df = pd.read_csv(IN_TRIAGE)

    # Only digitize where calibration is complete
    df = df[(df["digitizable"] == "yes") & (df["recommended_route"] == "digitize")].copy()

    # skip percent y-kind by default (common mismatch for your v1)
    if "y_kind" in df.columns:
        df = df[df["y_kind"] != "percent"].copy()

    df["plot_bbox_parsed"] = df["plot_bbox"].apply(parse_bbox)
    ok = (
        df["plot_bbox_parsed"].notna()
        & df["x_min"].notna() & df["x_max"].notna()
        & df["y_min"].notna() & df["y_max"].notna()
        & df["selected_image_path"].notna()
    )
    df_ok = df[ok].copy()

    print(f"[Digitize] candidates: {len(df)} | usable (bbox+ranges+image): {len(df_ok)}")

    if OUT_CURVES.exists():
        OUT_CURVES.unlink()

    end_rows: List[Dict[str, Any]] = []

    for _, r in df_ok.iterrows():
        doi = str(r["doi"]).strip()
        title = str(r.get("title", "") or "").strip()
        img_path = Path(str(r["selected_image_path"]))
        if not img_path.exists():
            end_rows.append({"doi": doi, "status": "fail_image_missing", "image_path": str(img_path)})
            continue

        bbox = r["plot_bbox_parsed"]
        x_min = float(r["x_min"]); x_max = float(r["x_max"])
        y_min = float(r["y_min"]); y_max = float(r["y_max"])
        x_unit = str(r.get("axes_x_unit", "") or str(r.get("x_unit", "") or ""))
        y_unit = str(r.get("axes_y_unit", "") or str(r.get("y_unit", "") or ""))
        y_kind = str(r.get("y_kind", "unknown") or "unknown")

        approx_k = r.get("approx_curves_count", None)
        try:
            approx_k = int(approx_k) if pd.notna(approx_k) else None
        except Exception:
            approx_k = None
        if approx_k is None or approx_k < 1:
            approx_k = 1
        approx_k = min(approx_k, 6)

        img = cv2.imread(str(img_path))
        if img is None:
            end_rows.append({"doi": doi, "status": "fail_image_read", "image_path": str(img_path)})
            continue

        crop = crop_by_bbox(img, bbox)
        h, w = crop.shape[:2]
        mask = extract_edge_mask(crop)

        ys, xs = np.where(mask > 0)
        if xs.size < MIN_EDGE_POINTS:
            end_rows.append({
                "doi": doi, "status": "fail_few_edges",
                "image_path": str(img_path), "edge_points": int(xs.size)
            })
            continue

        # sample HSV at edge pixels to separate curves by color
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hsv_pixels = hsv[ys, xs, :]  # Nx3

        # If only 1 curve or low color diversity, skip clustering
        curves = []
        if approx_k == 1:
            xpix, ypix = build_curve_from_points(xs, ys)
            if xpix.size >= 30:
                curves.append(("cluster_1", None, xs, ys, xpix, ypix))
        else:
            # kmeans clustering
            try:
                labels, centers = kmeans_cluster_colors(hsv_pixels, approx_k)
                for ki in range(approx_k):
                    idx = np.where(labels == ki)[0]
                    if idx.size < MIN_CLUSTER_POINTS:
                        continue
                    xs_k = xs[idx]; ys_k = ys[idx]
                    xpix, ypix = build_curve_from_points(xs_k, ys_k)
                    if xpix.size < 30:
                        continue
                    curves.append((f"cluster_{ki+1}", centers[ki], xs_k, ys_k, xpix, ypix))
            except Exception:
                # fallback to single-curve
                xpix, ypix = build_curve_from_points(xs, ys)
                if xpix.size >= 30:
                    curves.append(("cluster_1", None, xs, ys, xpix, ypix))

        if not curves:
            end_rows.append({"doi": doi, "status": "fail_no_curves", "image_path": str(img_path)})
            continue

        # For each curve, convert to data coords and compute Q_final at t_last=x_max
        for curve_id, center, xs_k, ys_k, xpix, ypix in curves:
            x_dat, y_dat = pixel_to_data(xpix, ypix, x_min, x_max, y_min, y_max, w, h)
            q_final = interp_at(x_dat, y_dat, x_max)
            t_last = x_max

            color_name = ""
            if center is not None:
                color_name = coarse_color_name(center)

            rec_curve = {
                "doi": doi,
                "title": title,
                "figure_id": str(r.get("figure_id", "") or ""),
                "page_number": int(r.get("page_number")) if pd.notna(r.get("page_number")) else None,
                "subplot": str(r.get("subplot", "") or ""),
                "image_path": str(img_path),
                "curve_id": curve_id,
                "curve_color": color_name,
                "x_unit": x_unit,
                "y_unit": y_unit,
                "y_kind": y_kind,
                "x_min": x_min, "x_max": x_max,
                "y_min": y_min, "y_max": y_max,
                "t_last": float(t_last),
                "q_final": float(q_final) if np.isfinite(q_final) else None,
                "curve_xy": [[float(a), float(b)] for a, b in zip(x_dat.tolist(), y_dat.tolist())],
            }

            with open(OUT_CURVES, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec_curve, ensure_ascii=False) + "\n")

            end_rows.append({
                "doi": doi,
                "title": title,
                "figure_id": rec_curve["figure_id"],
                "page_number": rec_curve["page_number"],
                "subplot": rec_curve["subplot"],
                "image_path": str(img_path),
                "status": "ok",
                "curve_id": curve_id,
                "curve_color": color_name,
                "endpoint_time": float(t_last),
                "endpoint_time_unit": x_unit,
                "endpoint_value": rec_curve["q_final"],
                "endpoint_unit": y_unit,
                "y_kind": y_kind
            })

    df_end = pd.DataFrame(end_rows)
    df_end.to_csv(OUT_ENDPTS, index=False, encoding="utf-8-sig")
    print("Done.")
    print("Saved curves:", OUT_CURVES)
    print("Saved endpoints:", OUT_ENDPTS)
    if "status" in df_end.columns:
        print(df_end["status"].value_counts(dropna=False))

if __name__ == "__main__":
    main()
