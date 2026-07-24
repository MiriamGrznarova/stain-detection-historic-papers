import os
import random
import cv2
import numpy as np

# =========================
# CONFIG
# =========================
BG_ROOT = r"splitted_dataset"
STAINS_ROOT = r"segmentated_samples"
OUT_ROOT = r"out_multi_stains_by_class1"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

MAX_TRIES_POS = 50
STAINS_PER_BG = (6, 22)
MASK_BLUR_SIGMA = 1.2

# =========================
# STAIN STATS
# =========================
# Tu necháš svoje ratio pre každú z 12 tried
STAIN_STATS = {
    ("bacterium", "sphingomonas aquatilis"): {"width_ratio": 0.14435519857206605, "height_ratio": 0.17103235747303544},
    ("bacterium", "rhodococcus degradans"): {"width_ratio": 0.16666666666666666, "height_ratio": 0.18541345659989728},
    ("bacterium", "micrococcus specius"): {"width_ratio": 0.2655064703257474, "height_ratio": 0.30148947098099643},
    ("bacterium", "curtobacterium flaccumfaciens"): {"width_ratio": 0.20571173583221777, "height_ratio": 0.23805855161787365},

    ("fungus", "epicoccum layunese"): {"width_ratio": 0.060240963855421686, "height_ratio": 0.06111967128916281},
    ("fungus", "gmb cladosporium pseudocladosporioides"): {"width_ratio": 0.2608210620258813, "height_ratio": 0.29301489470980996},
    ("bacterium", "exiguobacterium indicum gmb4"): {"width_ratio": 0.15930388219544847, "height_ratio": 0.18335901386748846},
    ("bacterium", "air-hu arthrobacter sp"): {"width_ratio": 0.24051762605979474, "height_ratio": 0.2593733949666153},

    ("fungus", "talaromyces purpureogenus"): {"width_ratio": 0.2568049977688532, "height_ratio": 0.2921160760143811},
    ("fungus", "penicilium mangini"): {"width_ratio": 0.22802320392681838, "height_ratio": 0.28197226502311246},
    ("fungus", "aspergillus ochraceus"): {"width_ratio": 0.20972780008924588, "height_ratio": 0.23369286081150487},
    ("fungus", "alternaria alternata"): {"width_ratio": 0.41231593038821956, "height_ratio": 0.46070878274268107},
}

# =========================
# CLASS IDS
# =========================
# Každej triede priradíme class_id
CLASS_MAP = {
    ("bacterium", "sphingomonas aquatilis"): 0,
    ("bacterium", "rhodococcus degradans"): 1,
    ("bacterium", "micrococcus specius"): 2,
    ("bacterium", "curtobacterium flaccumfaciens"): 3,
    ("fungus", "epicoccum layunese"): 4,
    ("fungus", "gmb cladosporium pseudocladosporioides"): 5,
    ("bacterium", "exiguobacterium indicum gmb4"): 6,
    ("bacterium", "air-hu arthrobacter sp"): 7,
    ("fungus", "talaromyces purpureogenus"): 8,
    ("fungus", "penicilium mangini"): 9,
    ("fungus", "aspergillus ochraceus"): 10,
    ("fungus", "alternaria alternata"): 11,
}

# =========================
# PAPER
# =========================
PAPER_SIZE_MM = 50.0
A4_W_MM = 210.0
A4_H_MM = 297.0

PAPER_RATIO_W = PAPER_SIZE_MM / A4_W_MM
PAPER_RATIO_H = PAPER_SIZE_MM / A4_H_MM


# =========================
# IO
# =========================
def list_images(folder):
    out = []
    for fn in os.listdir(folder):
        if fn.lower().endswith((".png", ".jpg", ".jpeg")):
            out.append(os.path.join(folder, fn))
    return sorted(out)


def read_bgr(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Cannot read: " + path)
    return img


def read_rgba(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("Cannot read: " + path)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)

    if img.shape[2] == 3:
        a = np.full((img.shape[0], img.shape[1], 1), 255, np.uint8)
        img = np.concatenate([img, a], axis=2)

    return img


def get_class_subclass_from_dir(path):
    norm = os.path.normpath(path)
    parts = norm.split(os.sep)
    subclass = parts[-1].strip().lower()
    cls = parts[-2].strip().lower()
    return cls, subclass


# =========================
# COLOR
# =========================
def bgr_to_cmyk(bgr_u8):
    bgr = bgr_u8.astype(np.float32) / 255.0
    b, g, r = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]

    c = 1.0 - r
    m = 1.0 - g
    y = 1.0 - b

    k = np.minimum(c, np.minimum(m, y))
    denom = np.clip(1.0 - k, 1e-6, 1.0)

    c = (c - k) / denom
    m = (m - k) / denom
    y = (y - k) / denom

    return np.stack([c, m, y, k], axis=2).astype(np.float32)


def cmyk_to_bgr(cmyk_f):
    c = np.clip(cmyk_f[:, :, 0], 0, 1)
    m = np.clip(cmyk_f[:, :, 1], 0, 1)
    y = np.clip(cmyk_f[:, :, 2], 0, 1)
    k = np.clip(cmyk_f[:, :, 3], 0, 1)

    r = (1 - c) * (1 - k)
    g = (1 - m) * (1 - k)
    b = (1 - y) * (1 - k)

    bgr = np.stack([b, g, r], axis=2)
    return np.clip(bgr * 255.0, 0, 255).astype(np.uint8)


def match_cmy_to_roi(stain_bgr_u8, roi_bgr_u8, mask01, strength=0.65):
    st_cmyk = bgr_to_cmyk(stain_bgr_u8)
    roi_cmyk = bgr_to_cmyk(roi_bgr_u8)

    w = np.clip(mask01.astype(np.float32), 0, 1) ** 1.7
    wsum = float(np.sum(w) + 1e-6)

    st_mean = np.sum(st_cmyk[:, :, :3] * w[:, :, None], axis=(0, 1)) / wsum
    roi_mean = np.sum(roi_cmyk[:, :, :3] * w[:, :, None], axis=(0, 1)) / wsum

    delta = (roi_mean - st_mean) * strength

    st_cmyk[:, :, 0] = np.clip(st_cmyk[:, :, 0] + delta[0] * w, 0, 1)
    st_cmyk[:, :, 1] = np.clip(st_cmyk[:, :, 1] + delta[1] * w, 0, 1)
    st_cmyk[:, :, 2] = np.clip(st_cmyk[:, :, 2] + delta[2] * w, 0, 1)

    return cmyk_to_bgr(st_cmyk)


# =========================
# MASKS
# =========================
def feather_mask(mask01, feather_px=12):
    m = (mask01 > 0).astype(np.uint8) * 255
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    feather = np.clip(dist / float(max(feather_px, 1)), 0, 1).astype(np.float32)
    return np.minimum(mask01.astype(np.float32), feather)


def smooth_ellipse_noise_mask(
    h, w,
    edge_noise=0.30,
    falloff_power=2.2,
    feather_blur=1.4,
    bias_strength=0.35,
    seed=None
):
    if seed is not None:
        rng = np.random.default_rng(seed)
        n = rng.random((h, w), dtype=np.float32)
        angle = rng.uniform(0, 2*np.pi)
    else:
        n = np.random.rand(h, w).astype(np.float32)
        angle = random.uniform(0, 2*np.pi)

    n = cv2.GaussianBlur(n, (0, 0), 4.5)
    n = (n - 0.5) * 2.0

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    a = w * 0.48
    b = h * 0.48

    r = np.sqrt(((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2)

    base = np.clip(1.0 - r, 0, 1)
    base = base ** falloff_power

    dx = (xx - cx) / max(a, 1e-6)
    dy = (yy - cy) / max(b, 1e-6)
    proj = dx * np.cos(angle) + dy * np.sin(angle)
    bias = np.clip((proj + 1.0) * 0.5, 0, 1)

    edge_weight = (1.0 - base)
    asym = (1.0 + bias_strength * (bias - 0.5) * 2.0 * edge_weight)

    mask = base * asym + n * edge_weight * edge_noise
    mask = cv2.GaussianBlur(mask, (0, 0), feather_blur)
    return np.clip(mask, 0, 1).astype(np.float32)

def crop_to_alpha_content(img_rgba, alpha_thresh=10):
    alpha = img_rgba[:, :, 3]
    ys, xs = np.where(alpha > alpha_thresh)

    if len(xs) == 0 or len(ys) == 0:
        return img_rgba

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    return img_rgba[y1:y2+1, x1:x2+1]

def build_stain_mask(st_rgba):
    bgr = st_rgba[:, :, :3].astype(np.uint8)
    a = st_rgba[:, :, 3].astype(np.float32) / 255.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    if a.mean() < 0.98:
        shape = (a > 0.10).astype(np.uint8) * 255
    else:
        bg_est = cv2.GaussianBlur(gray, (0, 0), 7.0)
        diff = np.abs(gray - bg_est)
        shape = (diff > 0.03).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    shape = cv2.morphologyEx(shape, cv2.MORPH_CLOSE, k, iterations=2)

    shape_f = shape.astype(np.float32) / 255.0
    if MASK_BLUR_SIGMA > 0:
        shape_f = cv2.GaussianBlur(shape_f, (0, 0), MASK_BLUR_SIGMA)

    bg_est = cv2.GaussianBlur(gray, (0, 0), 9.0)
    diff = np.abs(gray - bg_est)
    intensity = np.clip(diff / 0.25, 0, 1)

    m = shape_f * (0.55 + 0.45 * intensity)
    return np.clip(m, 0, 1).astype(np.float32)


def add_stain_texture(roi_bgr, stain_bgr, mask01, amount=0.6, blur_ksize=31):
    if blur_ksize % 2 == 0:
        blur_ksize += 1

    roi = roi_bgr.astype(np.float32)
    st = stain_bgr.astype(np.float32)

    st_blur = cv2.GaussianBlur(st, (blur_ksize, blur_ksize), 0)
    detail = st - st_blur

    m = mask01[:, :, None].astype(np.float32)
    out = roi + amount * detail * m

    return np.clip(out, 0, 255).astype(np.uint8)


# =========================
# GEOM
# =========================
def random_resize_by_paper_ratio(
    st_rgba,
    bg_h, bg_w,
    stain_width_ratio, stain_height_ratio,
    rand_range=(0.9, 1.1),
    keep_aspect=True
):
    sh, sw = st_rgba.shape[:2]

    target_w = bg_w * PAPER_RATIO_W * stain_width_ratio
    target_h = bg_h * PAPER_RATIO_H * stain_height_ratio

    rnd = random.uniform(*rand_range)
    target_w *= rnd
    target_h *= rnd

    if keep_aspect:
        scale = min(target_w / sw, target_h / sh)
        new_w = max(2, int(sw * scale))
        new_h = max(2, int(sh * scale))
    else:
        new_w = max(2, int(round(target_w)))
        new_h = max(2, int(round(target_h)))

    return cv2.resize(st_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)


def rotate_rgba(img_rgba):
    h, w = img_rgba.shape[:2]
    angle = random.uniform(0, 360)

    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)

    rotated = cv2.warpAffine(
        img_rgba,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255, 0)
    )

    return rotated


def overlaps(existing, x, y, w, h, pad=3):
    for (ex, ey, ew, eh) in existing:
        if not (x + w + pad < ex or x > ex + ew + pad or y + h + pad < ey or y > ey + eh + pad):
            return True
    return False


# =========================
# BLEND
# =========================
def paste_stain_cmyk_preserve_texture(bg_bgr, st_rgba, x, y, stains_dir):
    sh, sw = st_rgba.shape[:2]
    feather_px = int(np.clip(min(sh, sw) * 0.08, 6, 26))
    roi = bg_bgr[y:y+sh, x:x+sw].copy()

    stain_bgr = st_rgba[:, :, :3].astype(np.uint8)

    if "bacterium" in stains_dir.lower():
        shape_mask = smooth_ellipse_noise_mask(
            sh, sw,
            edge_noise=1.0,
            falloff_power=0.5,
            feather_blur=1.0,
            bias_strength=2.0
        )
    else:
        shape_mask = smooth_ellipse_noise_mask(
            sh, sw,
            edge_noise=1.0,
            falloff_power=0.9,
            feather_blur=1.0,
            bias_strength=3
        )

    mask = build_stain_mask(st_rgba)
    mask = np.clip(mask * shape_mask, 0, 1).astype(np.float32)

    mask = feather_mask(mask, feather_px)
    mask = np.clip(mask * 0.85, 0, 1).astype(np.float32)
    m3 = mask[:, :, None]

    stain_adapt = match_cmy_to_roi(stain_bgr, roi, mask, strength=0.60)

    st_gray = cv2.cvtColor(stain_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    absorb = 1.0 - (0.60 * mask * np.clip((1.0 - st_gray), 0, 1))
    roi_abs = roi.astype(np.float32) * absorb[:, :, None]

    out = roi_abs * (1 - m3) + stain_adapt.astype(np.float32) * m3
    out = np.clip(out, 0, 255).astype(np.uint8)

    k = int(max(21, (min(sh, sw) * 0.5) // 2 * 2 + 1))
    k = min(k, 61)

    out = add_stain_texture(out, stain_bgr, mask01=mask, amount=0.92, blur_ksize=k)
    bg_bgr[y:y+sh, x:x+sw] = out
    return bg_bgr, mask


# =========================
# OUTPUT
# =========================
def save_yolo_annotation(txt_path, class_id, boxes, img_w, img_h):
    lines = []
    for (bx1, by1, bx2, by2) in boxes:
        bw = max(1, bx2 - bx1)
        bh = max(1, by2 - by1)
        cx = bx1 + bw / 2.0
        cy = by1 + bh / 2.0

        x_center = cx / img_w
        y_center = cy / img_h
        width = bw / img_w
        height = bh / img_h

        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========================
# MAIN LOOP
# =========================
def get_all_stain_dirs(stains_root):
    stain_dirs = []
    for cls_name in os.listdir(stains_root):
        cls_path = os.path.join(stains_root, cls_name)
        if not os.path.isdir(cls_path):
            continue

        for subclass_name in os.listdir(cls_path):
            subclass_path = os.path.join(cls_path, subclass_name)
            if os.path.isdir(subclass_path):
                stain_dirs.append(subclass_path)

    return sorted(stain_dirs)


def process_one_stain_dir(stains_dir):
    cls_name, subclass_name = get_class_subclass_from_dir(stains_dir)
    key = (cls_name, subclass_name)

    if key not in STAIN_STATS:
        raise ValueError(f"Stats not found for {key}")

    if key not in CLASS_MAP:
        raise ValueError(f"Class ID not found for {key}")

    width_ratio = STAIN_STATS[key]["width_ratio"]
    height_ratio = STAIN_STATS[key]["height_ratio"]
    class_id = CLASS_MAP[key]

    folder_name = f"{cls_name}__{subclass_name}".replace(" ", "_")
    out_dir = os.path.join(OUT_ROOT, folder_name)

    out_images = os.path.join(out_dir, "images")
    out_masks = os.path.join(out_dir, "masks")
    out_labels = os.path.join(out_dir, "labels")

    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_masks, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    bg_dir_for_class = os.path.join(BG_ROOT, str(class_id))
    if not os.path.isdir(bg_dir_for_class):
        raise RuntimeError(f"No background folder for class {class_id}: {bg_dir_for_class}")

    bg_paths = list_images(bg_dir_for_class)
    stain_paths = list_images(stains_dir)

    if not bg_paths:
        raise RuntimeError(f"No backgrounds in {bg_dir_for_class}")
    if not stain_paths:
        raise RuntimeError("No stains in stain directory")

    print(f"\nProcessing: {key}")
    print(f"Class ID: {class_id}")
    print(f"Output dir: {out_dir}")

    for idx, bg_path in enumerate(bg_paths):
        bg = read_bgr(bg_path)
        H, W = bg.shape[:2]

        boxes = []
        global_mask = np.zeros((H, W), dtype=np.uint8)
        placed = []

        n_stains = random.randint(*STAINS_PER_BG)

        for _ in range(n_stains):
            st_path = random.choice(stain_paths)
            st = read_rgba(st_path)

            if cls_name == "fungus":
                rand_range = (1.5, 2)  # menšie
            else:
                rand_range = (0.9, 1.1)  # default
            st = random_resize_by_paper_ratio(
                st,
                bg_h=H,
                bg_w=W,
                stain_width_ratio=width_ratio,
                stain_height_ratio=height_ratio,
                rand_range=rand_range,
                keep_aspect=True
            )
            st = rotate_rgba(st)
            st = crop_to_alpha_content(st, alpha_thresh=10)

            sh, sw = st.shape[:2]
            if sh >= H or sw >= W:
                continue

            ok = False
            for _try in range(MAX_TRIES_POS):
                x = random.randint(0, W - sw)
                y = random.randint(0, H - sh)
                if not overlaps(placed, x, y, sw, sh, pad=5):
                    ok = True
                    break
            if not ok:
                continue

            bg, l_mask = paste_stain_cmyk_preserve_texture(bg, st, x, y, stains_dir=stains_dir)
            l_mask_u8 = (l_mask * 255).astype(np.uint8)

            y1, x1 = y + sh, x + sw
            global_mask[y:y1, x:x1] = np.maximum(global_mask[y:y1, x:x1], l_mask_u8)

            ys, xs = np.where(l_mask_u8 > 10)
            if len(xs) > 0 and len(ys) > 0:
                bx1 = x + int(xs.min())
                by1 = y + int(ys.min())
                bx2 = x + int(xs.max())
                by2 = y + int(ys.max())
                boxes.append((bx1, by1, bx2, by2))

            placed.append((x, y, sw, sh))

        base = os.path.splitext(os.path.basename(bg_path))[0]

        cv2.imwrite(os.path.join(out_images, f"{base}.png"), bg)
        cv2.imwrite(os.path.join(out_masks, f"{base}_mask.png"), global_mask)
        save_yolo_annotation(
            os.path.join(out_labels, f"{base}.txt"),
            class_id,
            boxes,
            W, H
        )

        print(f"[{idx+1}/{len(bg_paths)}] saved {base} with {len(boxes)} stains")


def main():
    # process_one_stain_dir(r"segmentated_samples\fungus\alternaria alternata")
    # process_one_stain_dir(r"segmentated_samples\fungus\aspergillus ochraceus")
    process_one_stain_dir(r"segmentated_samples\fungus\epicoccum layunese")
    # process_one_stain_dir(r"segmentated_samples\fungus\gmb cladosporium pseudocladosporioides")
    # process_one_stain_dir(r"segmentated_samples\fungus\penicilium mangini")
    # process_one_stain_dir(r"segmentated_samples\fungus\talaromyces purpureogenus")


if __name__ == "__main__":
    main()