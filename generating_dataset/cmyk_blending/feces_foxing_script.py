import os
import random
import cv2
import numpy as np

# =========================
# CONFIG
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BG_DIR = r"C:\Users\miria\Desktop\bakalarske_studium\bakalarka_statnice\new_dataset\large_newspaper"

FOXING_DIR = r"C:\Users\miria\Desktop\bakalarske_studium\bakalarka_statnice\new_dataset\segmentated_samples\foxing"
FECES_DIR = r"C:\Users\miria\Desktop\bakalarske_studium\bakalarka_statnice\new_dataset\segmentated_samples\vykaly"

OUT_ROOT = r"out_foxing_feces"

MAX_TRIES_POS = 50
MASK_BLUR_SIGMA = 1.2

# koľko škvŕn na background
FOXING_PER_BG = (6, 18)
FECES_PER_BG = (6, 18)
COMBO_FOXING_PER_BG = (3, 10)
COMBO_FECES_PER_BG = (3, 10)

# =========================
# CLASS IDS
# =========================
CLASS_IDS = {
    "foxing": 0,
    "feces": 5
}

# =========================
# STAIN STATS
# =========================
# Tu si nastav svoje ratio podľa papierika 5x5 cm.
# Zatiaľ sú to placeholdery - uprav si ich podľa svojich reálnych metrík.
STAIN_STATS = {
    ("foxing", "foxing"): {
        "width_ratio": 0.18,
        "height_ratio": 0.16
    },
    ("feces", "vykaly"): {
        "width_ratio": 0.035,
        "height_ratio": 0.028
    },
}

# =========================
# PAPER RATIO
# =========================
PAPER_SIZE_MM = 50.0
A4_W_MM = 210.0
A4_H_MM = 297.0

PAPER_RATIO_W = PAPER_SIZE_MM / A4_W_MM
PAPER_RATIO_H = PAPER_SIZE_MM / A4_H_MM

TINT_MAPPING = {
    "foxing": np.array([20, 95, 170], dtype=np.float32),
    "feces": np.array([8, 18, 30], dtype=np.float32)
}
# =========================
# COLOR
# =========================

def tint_stain_bgr(stain_bgr, mask01, tint_bgr, amount=0.35):
    """
    Jemne prifarbi škvrnu na zvolený BGR odtieň.
    amount: 0..1
    """
    stain_f = stain_bgr.astype(np.float32)
    tint_img = np.ones_like(stain_f, dtype=np.float32) * tint_bgr.reshape(1, 1, 3)

    m3 = mask01[:, :, None].astype(np.float32)
    out = stain_f * (1.0 - amount * m3) + tint_img * (amount * m3)

    return np.clip(out, 0, 255).astype(np.uint8)

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

    bgr = img_rgba[:, :, :3]
    a = img_rgba[:, :, 3]

    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)

    bgr_r = cv2.warpAffine(
        bgr, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    a_r = cv2.warpAffine(
        a, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return np.dstack([bgr_r, a_r])


def overlaps(existing, x, y, w, h, pad=3):
    for (ex, ey, ew, eh) in existing:
        if not (x + w + pad < ex or x > ex + ew + pad or y + h + pad < ey or y > ey + eh + pad):
            return True
    return False


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
        angle = rng.uniform(0, 2 * np.pi)
    else:
        n = np.random.rand(h, w).astype(np.float32)
        angle = random.uniform(0, 2 * np.pi)

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


def build_stain_mask(st_rgba):
    bgr = st_rgba[:, :, :3].astype(np.uint8)
    a = st_rgba[:, :, 3].astype(np.float32) / 255.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    if a.mean() < 0.98:
        shape = (a > 0.02).astype(np.uint8) * 255
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
# BLEND
# =========================
def paste_stain_cmyk_preserve_texture(
    bg_bgr, st_rgba, x, y, stain_kind,
    opacity=0.85,
    cmy_match_strength=0.60,
    texture_amount=0.92,
    k_absorb_strength=0.60,
):
    sh, sw = st_rgba.shape[:2]
    feather_px = int(np.clip(min(sh, sw) * 0.08, 4, 26))
    roi = bg_bgr[y:y+sh, x:x+sw].copy()

    stain_bgr = st_rgba[:, :, :3].astype(np.uint8)

    if stain_kind == "foxing":
        shape_mask = smooth_ellipse_noise_mask(
            sh, sw,
            edge_noise=0.9,
            falloff_power=1.1,
            feather_blur=4.0,
            bias_strength=2.0
        )
        opacity = 0.78
        tint_amount = 0.30
        cmy_match_strength = 0.22
        k_absorb_strength = 0.60

    else:  # feces
        shape_mask = smooth_ellipse_noise_mask(
            sh, sw,
            edge_noise=0.7,
            falloff_power=1.5,
            feather_blur=1.8,
            bias_strength=1.2
        )
        opacity = 0.96
        tint_amount = 0.08
        cmy_match_strength = 0.10
        k_absorb_strength = 1.00

    mask = build_stain_mask(st_rgba)
    mask = np.clip(mask * shape_mask, 0, 1).astype(np.float32)

    mask = feather_mask(mask, feather_px)
    mask = np.clip(mask * opacity, 0, 1).astype(np.float32)
    m3 = mask[:, :, None]

    # prisposobenie farby papieru
    stain_adapt = match_cmy_to_roi(stain_bgr, roi, mask, strength=cmy_match_strength)

    # tonalita
    tint_bgr = TINT_MAPPING[stain_kind]
    stain_adapt = tint_stain_bgr(stain_adapt, mask, tint_bgr, amount=tint_amount)

    # silnejsie stmavenie len pre feces
    if stain_kind == "feces":
        stain_adapt = np.clip(stain_adapt.astype(np.float32) * 0.25, 0, 255).astype(np.uint8)

    st_gray = cv2.cvtColor(stain_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    if stain_kind == "feces":
        absorb = 1.0 - (1.6 * k_absorb_strength * mask * np.clip((1.0 - st_gray), 0, 1))
    else:
        absorb = 1.0 - (k_absorb_strength * mask * np.clip((1.0 - st_gray), 0, 1))
    roi_abs = roi.astype(np.float32) * absorb[:, :, None]

    if stain_kind == "foxing":
        stain_norm = stain_adapt.astype(np.float32) / 255.0
        roi_norm = roi.astype(np.float32) / 255.0

        stain_norm[:, :, 2] *= 1.15
        stain_norm[:, :, 1] *= 1.05

        out = roi_norm * (1 - m3 + m3 * stain_norm)
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)

    else:
        # zakladne vlozenie feces
        out = roi_abs * (1 - m3) + stain_adapt.astype(np.float32) * m3

        # extra blending len na hranach
        core_mask = feather_mask(mask, feather_px=max(2, feather_px // 2))
        edge_zone = np.clip(mask - core_mask, 0, 1).astype(np.float32)
        edge3 = edge_zone[:, :, None]

        # na okrajoch viac miesaj s papierom
        out = out * (1 - 0.35 * edge3) + roi.astype(np.float32) * (0.35 * edge3)

        out = np.clip(out, 0, 255).astype(np.uint8)

    out = np.clip(out, 0, 255).astype(np.uint8)

    k = int(max(21, (min(sh, sw) * 0.5) // 2 * 2 + 1))
    k = min(k, 61)

    out = add_stain_texture(out, stain_bgr, mask01=mask, amount=texture_amount, blur_ksize=k)
    bg_bgr[y:y+sh, x:x+sw] = out
    return bg_bgr, mask

# =========================
# YOLO
# =========================
def save_yolo_annotations(txt_path, annotations, img_w, img_h):
    lines = []
    for class_id, bx1, by1, bx2, by2 in annotations:
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
# CORE
# =========================
def place_stains_of_one_class(
    bg,
    global_mask,
    placed,
    annotations,
    stain_paths,
    stain_kind,
    stain_width_ratio,
    stain_height_ratio,
    stains_per_bg,
):
    H, W = bg.shape[:2]
    n_stains = random.randint(*stains_per_bg)

    for _ in range(n_stains):
        st_path = random.choice(stain_paths)
        st = read_rgba(st_path)

        if stain_kind == "foxing":
            rand_range = (1.2, 2.0)
        else:
            rand_range = (0.7, 1)

        st = random_resize_by_paper_ratio(
            st,
            bg_h=H,
            bg_w=W,
            stain_width_ratio=stain_width_ratio,
            stain_height_ratio=stain_height_ratio,
            rand_range=rand_range,
            keep_aspect=True
        )
        st = rotate_rgba(st)

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

        bg, l_mask = paste_stain_cmyk_preserve_texture(bg, st, x, y, stain_kind=stain_kind)
        l_mask_u8 = (l_mask * 255).astype(np.uint8)

        y1, x1 = y + sh, x + sw
        global_mask[y:y1, x:x1] = np.maximum(global_mask[y:y1, x:x1], l_mask_u8)

        ys, xs = np.where(l_mask_u8 > 10)
        if len(xs) > 0 and len(ys) > 0:
            bx1 = x + int(xs.min())
            by1 = y + int(ys.min())
            bx2 = x + int(xs.max())
            by2 = y + int(ys.max())

            class_id = CLASS_IDS["foxing"] if stain_kind == "foxing" else CLASS_IDS["feces"]
            annotations.append((class_id, bx1, by1, bx2, by2))

        placed.append((x, y, sw, sh))

    return bg, global_mask, placed, annotations


def process_dataset(mode_name, use_foxing, use_feces):
    out_dir = os.path.join(OUT_ROOT, mode_name)

    out_images = os.path.join(out_dir, "images")
    out_masks = os.path.join(out_dir, "masks")
    out_images_bb = os.path.join(out_dir, "images_bb")
    out_labels = os.path.join(out_dir, "labels")
    out_boxes = os.path.join(out_dir, "boxes_xyxy")

    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_masks, exist_ok=True)
    os.makedirs(out_images_bb, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)
    os.makedirs(out_boxes, exist_ok=True)

    bg_paths = list_images(BG_DIR)
    foxing_paths = list_images(FOXING_DIR) if use_foxing else []
    feces_paths = list_images(FECES_DIR) if use_feces else []

    if not bg_paths:
        raise RuntimeError("No backgrounds in BG_DIR")
    if use_foxing and not foxing_paths:
        raise RuntimeError("No foxing images found")
    if use_feces and not feces_paths:
        raise RuntimeError("No feces images found")

    foxing_ratios = STAIN_STATS[("foxing", "foxing")]
    feces_ratios = STAIN_STATS[("feces", "vykaly")]

    print(f"\n=== Processing dataset: {mode_name} ===")

    for idx, bg_path in enumerate(bg_paths):
        bg = read_bgr(bg_path)
        H, W = bg.shape[:2]

        global_mask = np.zeros((H, W), dtype=np.uint8)
        placed = []
        annotations = []

        if use_foxing:
            stains_per_bg = FOXING_PER_BG if mode_name == "foxing_only" else COMBO_FOXING_PER_BG
            bg, global_mask, placed, annotations = place_stains_of_one_class(
                bg, global_mask, placed, annotations,
                foxing_paths, "foxing",
                foxing_ratios["width_ratio"], foxing_ratios["height_ratio"],
                stains_per_bg
            )

        if use_feces:
            stains_per_bg = FECES_PER_BG if mode_name == "feces_only" else COMBO_FECES_PER_BG
            bg, global_mask, placed, annotations = place_stains_of_one_class(
                bg, global_mask, placed, annotations,
                feces_paths, "feces",
                feces_ratios["width_ratio"], feces_ratios["height_ratio"],
                stains_per_bg
            )

        base = os.path.splitext(os.path.basename(bg_path))[0]

        bg_bb = bg.copy()
        for class_id, bx1, by1, bx2, by2 in annotations:
            if class_id == 0:
                color = (0, 165, 255)  # foxing
            else:
                color = (0, 0, 255)    # feces
            cv2.rectangle(bg_bb, (bx1, by1), (bx2, by2), color, 2)

        cv2.imwrite(os.path.join(out_images, f"{base}.png"), bg)
        cv2.imwrite(os.path.join(out_masks, f"{base}_mask.png"), global_mask)
        cv2.imwrite(os.path.join(out_images_bb, f"{base}_bb.png"), bg_bb)

        save_yolo_annotations(
            os.path.join(out_labels, f"{base}.txt"),
            annotations,
            W, H
        )

        with open(os.path.join(out_boxes, f"{base}.txt"), "w", encoding="utf-8") as f:
            for class_id, bx1, by1, bx2, by2 in annotations:
                f.write(f"{class_id},{bx1},{by1},{bx2},{by2}\n")

        print(f"[{idx + 1}/{len(bg_paths)}] saved {base} with {len(annotations)} stains")


def main():
    process_dataset(mode_name="foxing_only", use_foxing=True, use_feces=False)
    process_dataset(mode_name="feces_only", use_foxing=False, use_feces=True)
    process_dataset(mode_name="combo", use_foxing=True, use_feces=True)
    print("\nDONE.")


if __name__ == "__main__":
    main()