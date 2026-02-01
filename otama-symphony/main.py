# =========================================================
# 🎹 Otama Symphony - Piano + Song Mode (Open-source friendly main.py)
#
# ✅ この main.py の方針
#  - 画像や音の“中身”に依存しない（fish / river などの名前を排除）
#  - 役割ベース（background / key / normal_object / rare_object / hit_effect）で扱う
#  - paths は config.json だけ編集すれば差し替えできる
#  - settings_dir も config.json の指定を使う
#
# 必要ファイル:
#   - main.py と同じ階層に config.json
#   - config.json で指定した assets/ 以下の画像・音
# =========================================================

import os, json, time, math, random
import pathlib
from collections import deque

import cv2
import numpy as np
import pygame
import tkinter as tk

# =========================================================
# ✅ ノート定義（サウンドロードより前に必要）
# =========================================================
NOTE_ORDER  = ["C4","D4","E4","F4","G4","A4","B4","C5"]
NOTE_LABELS = ["C4","D4","E4","F4","G4","A4","B4","C5"]

# =========================================================
# ✅ config.json を読み込む（初心者はここだけ編集すればOK）
# =========================================================
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.json"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"config.json が見つかりません: {CONFIG_PATH}\n"
        f"→ main.py と同じフォルダに config.json を置いてください。"
    )

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

def p(path_str: str) -> str:
    """config内のパス（相対/絶対）を絶対パスに変換"""
    ps = pathlib.Path(path_str)
    if ps.is_absolute():
        return str(ps.resolve())
    return str((SCRIPT_DIR / ps).resolve())

print("✅ config.json を読み込みました")

# =========================================================
# ✅ 役割ベースのアセットパス（中身に依存しない命名）
# =========================================================
BACKGROUND_IMG_PATH = p(CONFIG["paths"]["images"]["background"])
KEY_IMG_PATH        = p(CONFIG["paths"]["images"]["key"])

NORMAL_LEFT_PATH    = p(CONFIG["paths"]["images"]["normal_left"])
NORMAL_RIGHT_PATH   = p(CONFIG["paths"]["images"]["normal_right"])
RARE_LEFT_PATH      = p(CONFIG["paths"]["images"]["rare_left"])
RARE_RIGHT_PATH     = p(CONFIG["paths"]["images"]["rare_right"])

HIT_EFFECT_PATH     = p(CONFIG["paths"]["images"]["hit_effect"])

BGM_PATH            = p(CONFIG["paths"]["sounds"]["bgm"])
NOTES_DIR           = p(CONFIG["paths"]["sounds"]["notes_dir"])
HIT_SFX_PATH        = p(CONFIG["paths"]["sounds"]["hit"])
RARE_HIT_SFX_PATH   = p(CONFIG["paths"]["sounds"]["rare_hit"])

SETTINGS_DIR        = p(CONFIG["paths"]["settings_dir"])
os.makedirs(SETTINGS_DIR, exist_ok=True)

# =========================================================
# 🎵 Audio 初期化（config ベース）
# =========================================================
pygame.mixer.init()

notes_volume = float(CONFIG.get("audio", {}).get("notes_volume", 0.9))
sfx_volume   = float(CONFIG.get("audio", {}).get("sfx_volume", 0.9))
bgm_volume   = float(CONFIG.get("audio", {}).get("bgm_volume", 0.5))

SOUNDS = {}
for n in NOTE_ORDER:
    wav_path = os.path.join(NOTES_DIR, f"{n}.wav")
    if os.path.exists(wav_path):
        try:
            SOUNDS[n] = pygame.mixer.Sound(wav_path)
            SOUNDS[n].set_volume(notes_volume)
        except Exception as e:
            print(f"⚠ notes load failed: {wav_path} -> {e}")
    else:
        print(f"ℹ️ note missing: {wav_path}")

HIT_SFX = pygame.mixer.Sound(HIT_SFX_PATH) if os.path.exists(HIT_SFX_PATH) else None
RARE_HIT_SFX = pygame.mixer.Sound(RARE_HIT_SFX_PATH) if os.path.exists(RARE_HIT_SFX_PATH) else None
if HIT_SFX: HIT_SFX.set_volume(sfx_volume)
if RARE_HIT_SFX: RARE_HIT_SFX.set_volume(sfx_volume)

if os.path.exists(BGM_PATH):
    pygame.mixer.music.set_volume(bgm_volume)
    pygame.mixer.music.load(BGM_PATH)
    pygame.mixer.music.play(-1)
else:
    print(f"⚠ bgm not found: {BGM_PATH}")

# =========================================================
# 🌊 Ripple クラス（背景上のエフェクトとして汎用）
# =========================================================
class Ripple:
    def __init__(self, x, y, start_time):
        self.x = int(x)
        self.y = int(y)
        self.start_time = start_time
        self.duration = random.uniform(1.6, 2.2)
        self.speed = random.uniform(170, 210)
        self.spacing = random.uniform(11, 15)
        self.irregular = random.uniform(0.02, 0.05)
        self.color = (250, 255, 255)  # B,G,R

    def is_alive(self, now):
        return (now - self.start_time) < self.duration

    def draw(self, frame, now):
        t = now - self.start_time
        if t < 0:
            return

        life = t / self.duration
        if life >= 1:
            return

        base_radius = self.speed * t
        fade = (1 - life) ** 2

        overlay = np.zeros_like(frame)
        n = 0
        while True:
            r = base_radius - n * self.spacing
            if r <= 0:
                break
            local_alpha = fade * np.exp(-n * 0.22)
            if local_alpha < 0.02:
                break

            thickness = max(1, int(2.5 - n * 0.08))
            jitter = random.uniform(-1.0, 1.5)
            radius = max(1, r + jitter)

            scale_x = 1.0 + self.irregular
            scale_y = 1.0 - self.irregular
            axes = (int(radius * scale_x), int(radius * scale_y))

            cv2.ellipse(
                overlay,
                (self.x, self.y),
                axes,
                0, 0, 360,
                self.color,
                thickness,
                lineType=cv2.LINE_AA
            )
            n += 1

        cv2.addWeighted(overlay, 0.35 * fade, frame, 1.0, 0, frame)

        center_r = int(4 * (1 - life))
        if center_r > 0:
            center_overlay = np.zeros_like(frame)
            cv2.circle(center_overlay, (self.x, self.y), center_r,
                       (255, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.addWeighted(center_overlay, 0.45 * fade, frame, 1.0, 0, frame)

class EffectFlash:
    def __init__(self, x, y, start_time, scale=0.25):
        self.x = int(x)
        self.y = int(y)
        self.start_time = start_time
        self.duration = 0.6
        self.fade_max = 0.95
        self.scale = float(scale)

    def is_alive(self, now):
        return (now - self.start_time) < self.duration

    def draw(self, frame, now, effect_img):
        if effect_img is None:
            return

        t = now - self.start_time
        if t < 0:
            return
        life = t / self.duration
        if life >= 1:
            return

        # 0→1→0 の山形フェード
        a = 1.0 - abs(2.0*life - 1.0)
        fade = self.fade_max * (a ** 1.2)

        # ✅ ここで縮小してから描画（小さすぎ防止も）
        h0, w0 = effect_img.shape[:2]
        w = max(12, int(w0 * self.scale))
        h = max(12, int(h0 * self.scale))
        eff = cv2.resize(effect_img, (w, h), interpolation=cv2.INTER_AREA)

        blend_rgba(frame, eff, self.x, self.y, fade)


# =========================================================
# 🖥 Screen size
# =========================================================
def get_screen_size():
    root = tk.Tk()
    root.withdraw()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.destroy()
    return int(sw), int(sh)

screen_w, screen_h = get_screen_size()
print(f"🖥 Screen size: {screen_w} x {screen_h}")

# =========================================================
# 🖼 背景・鍵盤画像ロード（役割ベース）
# =========================================================
bg0 = cv2.imread(BACKGROUND_IMG_PATH)
if bg0 is None:
    raise FileNotFoundError(f"❌ background image が見つかりません: {BACKGROUND_IMG_PATH}")

bg0 = cv2.resize(bg0, (screen_w, screen_h), interpolation=cv2.INTER_AREA)
bg  = bg0.copy()
proj_h, proj_w = bg.shape[:2]

key_img_raw = cv2.imread(KEY_IMG_PATH, cv2.IMREAD_COLOR)
if key_img_raw is None:
    print(f"⚠ key image が見つかりません: {KEY_IMG_PATH}")
    key_img = None
else:
    kh0, kw0 = key_img_raw.shape[:2]
    key_scale = min((proj_h * 0.90) / kh0, 1.0)
    new_w = int(kw0 * key_scale)
    new_h = int(kh0 * key_scale)
    key_img = cv2.resize(key_img_raw, (new_w, new_h), interpolation=cv2.INTER_AREA)

# =========================================================
# 🧩 通常/レア オブジェクト画像ロード（中身不問）
# =========================================================
normal_left_raw  = cv2.imread(NORMAL_LEFT_PATH,  cv2.IMREAD_UNCHANGED)
normal_right_raw = cv2.imread(NORMAL_RIGHT_PATH, cv2.IMREAD_UNCHANGED)
if normal_left_raw is None or normal_right_raw is None:
    print("⚠ normal_left/right が読み込めません。config.json を確認してください。")
    normal_left = None
    normal_right = None
else:
    normal_left = normal_left_raw
    normal_right = normal_right_raw

rare_left_raw  = cv2.imread(RARE_LEFT_PATH,  cv2.IMREAD_UNCHANGED)
rare_right_raw = cv2.imread(RARE_RIGHT_PATH, cv2.IMREAD_UNCHANGED)
if rare_left_raw is None or rare_right_raw is None:
    print("ℹ️ rare_left/right が読み込めません（レア出現は無効になります）")
    rare_left = None
    rare_right = None
else:
    rare_left = rare_left_raw
    rare_right = rare_right_raw

hit_effect_raw = cv2.imread(HIT_EFFECT_PATH, cv2.IMREAD_UNCHANGED)
if hit_effect_raw is None:
    hit_effect = None
else:
    hit_effect = hit_effect_raw

# =========================================================
# 💾 設定保存 / 読み込み
# =========================================================
def settings_path(name="grid_config"):
    return os.path.join(SETTINGS_DIR, f"{name}.json")

# 初期値（あとで load で上書きされる）
default_abs_width   = proj_w / 8.0
abs_widths          = [default_abs_width]*8
grid_offset_x       = 0.0
grid_scale_x        = 1.00
grid_height_scale   = 1.00
shift_x             = +20

height_corr_enabled = False
height_corr_ratio   = 0.00

selected_idx = 0
show_proj_grid   = True
show_proj_labels = True

def save_named_settings(name="grid_config"):
    s = {
        "abs_widths": abs_widths,
        "grid_offset_x": grid_offset_x,
        "grid_scale_x": grid_scale_x,
        "height_scale": grid_height_scale,
        "shift_x": shift_x,
        "height_correction_enabled": height_corr_enabled,
        "height_correction_ratio": height_corr_ratio,
    }
    try:
        with open(settings_path(name), "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=2)
        print(f"💾 設定を保存: {settings_path(name)}")
    except Exception as e:
        print("⚠ 設定保存エラー:", e)

def load_named_settings(name="grid_config"):
    global abs_widths, grid_offset_x, grid_scale_x
    global grid_height_scale, shift_x, height_corr_enabled, height_corr_ratio
    path = settings_path(name)
    if not os.path.exists(path):
        print(f"ℹ️ 設定ファイル {path} が見つかりません。")
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            s = json.load(f)
        if "abs_widths" in s and len(s["abs_widths"])==8:
            abs_widths[:] = s["abs_widths"]
        elif "note_widths" in s and len(s["note_widths"])==8:
            base = proj_w/8.0
            abs_widths[:] = [base*r for r in s["note_widths"]]
        grid_offset_x       = float(s.get("grid_offset_x", grid_offset_x))
        grid_scale_x        = float(s.get("grid_scale_x",  grid_scale_x))
        grid_height_scale   = float(s.get("height_scale", grid_height_scale))
        shift_x             = int(s.get("shift_x", shift_x))
        height_corr_enabled = bool(s.get("height_correction_enabled", height_corr_enabled))
        height_corr_ratio   = float(s.get("height_correction_ratio", height_corr_ratio))
        print(f"✅ 設定を読み込み: {path}")
    except Exception as e:
        print("⚠ 設定読込エラー:", e)

load_named_settings()

# =========================================================
# 🎹 鍵盤レーンのフェード管理（HALOなし）
# =========================================================
key_opacity = [0.0] * 8
key_state   = [0]   * 8

KEY_FADE_IN_SPEED  = 1.6
KEY_FADE_OUT_SPEED = 0.20
KEY_TARGET_ON   = 0.90
KEY_TARGET_IDLE = 0.50

def activate_key_lane(idx):
    if 0 <= idx < 8:
        if key_state[idx] in (0,2):
            key_state[idx] = 1

def update_key_lanes(dt):
    for i in range(8):
        if key_state[i] == 1:
            key_opacity[i] += dt * KEY_FADE_IN_SPEED
            if key_opacity[i] >= KEY_TARGET_ON:
                key_opacity[i] = KEY_TARGET_ON
                key_state[i] = 2
        elif key_state[i] == 2:
            if key_opacity[i] > KEY_TARGET_IDLE:
                key_opacity[i] -= dt * KEY_FADE_OUT_SPEED
                if key_opacity[i] < KEY_TARGET_IDLE:
                    key_opacity[i] = KEY_TARGET_IDLE

def draw_keys(proj_img):
    if key_img is None:
        return proj_img
    out = proj_img.copy()
    kh, kw = key_img.shape[:2]
    lane_w = proj_w / 8.0
    y1 = int((proj_h - kh) / 2)
    y2 = y1 + kh

    for i in range(8):
        alpha = key_opacity[i]
        if alpha <= 0:
            continue

        center_x = int((i + 0.5) * lane_w)
        x1 = int(center_x - kw/2)
        x2 = x1 + kw

        rx1, ry1 = max(0, x1), max(0, y1)
        rx2, ry2 = min(proj_w, x2), min(proj_h, y2)

        sx1, sy1 = rx1 - x1, ry1 - y1
        sx2, sy2 = sx1 + (rx2 - rx1), sy1 + (ry2 - ry1)

        kb_crop = key_img[sy1:sy2, sx1:sx2]
        roi     = out[ry1:ry2, rx1:rx2]
        cv2.addWeighted(kb_crop, alpha, roi, 1-alpha, 0, roi)

    return out

# =========================================================
# 🧮 グリッド関連
# =========================================================
def grid_boundaries_expand(offset_x, scale_x, abs_ws):
    xs = [offset_x]
    acc = offset_x
    for w in abs_ws:
        acc += w * scale_x
        xs.append(acc)
    return xs

def grid_boundaries_canonical(offset_x):
    xs = [offset_x]
    cell = proj_w / 8.0
    for _ in range(8):
        xs.append(xs[-1] + cell)
    return xs

def apply_height_correction(px, py):
    if not height_corr_enabled or height_corr_ratio==0.0:
        return px
    y_norm = (py - proj_h*0.5) / proj_h
    approx_cell_w = (sum(abs_widths)/8.0) * grid_scale_x
    return int(px + height_corr_ratio * y_norm * approx_cell_w * 5.0)

def build_base():
    out = bg.copy()
    min_band = 40
    gh = int(proj_h * grid_height_scale)
    gh = max(min_band, min(proj_h, gh))
    pad = (proj_h - gh)//2
    top_y    = pad
    bottom_y = proj_h - pad - 1

    xs_expand    = grid_boundaries_expand(grid_offset_x, grid_scale_x, abs_widths)
    xs_canonical = grid_boundaries_canonical(0.0)

    if show_proj_grid or show_proj_labels:
        for i in range(len(NOTE_ORDER)):
            gx = int(xs_expand[i])
            if show_proj_grid:
                cv2.line(out,(gx, top_y),(gx, bottom_y),(180,180,180),2)
            if show_proj_labels:
                lx = min(max(gx+12, 10), proj_w-80)
                cv2.putText(out, NOTE_LABELS[i], (lx, max(40, top_y+40)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60,60,60), 2)
        if show_proj_grid:
            gx_end = int(xs_expand[-1])
            cv2.line(out,(gx_end, top_y),(gx_end, bottom_y),(180,180,180),2)

    return out, xs_expand, xs_canonical, top_y, bottom_y

# =========================================================
# 🎮 モード管理
# =========================================================
MODE = 0  # 0: Piano, 1: Song
MODE_NAMES = ["Piano", "Song"]

# =========================================================
# 🎵 Song データ（3曲）
# =========================================================
SONG_TWINKLE = [
    "C4","C4","G4","G4","A4","A4","G4",
    "F4","F4","E4","E4","D4","D4","C4",
    "G4","G4","F4","F4","E4","E4","D4",
    "G4","G4","F4","F4","E4","E4","D4",
    "C4","C4","G4","G4","A4","A4","G4",
    "F4","F4","E4","E4","D4","D4","C4"
]
SONG_FROG = [
    "C4","D4","E4","F4",
    "E4","D4","C4",
    "E4","F4","G4",
    "A4","G4","F4","E4",
    "C4","C4","C4","C4","C4","C4",
    "D4","D4",
    "E4","E4",
    "F4","F4",
    "E4","D4","C4"
]
SONG_MARY = [
    "E4","D4","C4","D4",
    "E4","E4","E4",
    "D4","D4","D4",
    "E4","G4","G4",
    "E4","D4","C4","D4",
    "E4","E4","E4",
    "E4","D4","D4",
    "E4","D4","C4"
]
SONGS = {
    "twinkle": {"name": "きらきら星",        "notes": SONG_TWINKLE},
    "frog":    {"name": "かえるのうた",      "notes": SONG_FROG},
    "jingle":  {"name": "メリーさんのひつじ", "notes": SONG_MARY}
}
current_song_id = "twinkle"
song_index = 0

# =========================================================
# 🎥 カメラ初期化
# =========================================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, -6)
print("🎥 カメラ初期化完了")

# =========================================================
# 📐 変換行列を作る関数（M を使う前に必要）
# =========================================================
proj_pts = np.float32([[0,0],[proj_w,0],[proj_w,proj_h],[0,proj_h]])

def compute_M(cam_pts, shift_x):
    center = np.mean(cam_pts, axis=0)
    scale  = 1.08
    cam2   = (cam_pts - center) * scale + center
    cam2[:, 0] += shift_x
    return cv2.getPerspectiveTransform(cam2, proj_pts)

# =========================================================
# 📐 4点キャリブレーション（いつでも再実行できる）
# =========================================================
def run_calibration(cap):
    print("📐 投影床をクリック（左上→右上→右下→左下） / ESCで中断")
    clicked_points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
            clicked_points.append([x, y])
            print(f"✅ Point {len(clicked_points)}: ({x},{y})")

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("⚠ カメラフレーム取得失敗")
            break

        tmp = frame.copy()
        for p0 in clicked_points:
            cv2.circle(tmp, tuple(p0), 5, (0, 255, 255), -1)

        cv2.putText(tmp, "Click 4 points: TL->TR->BR->BL (ESC cancel)",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Calibration", tmp)
        cv2.setMouseCallback("Calibration", click_event)

        k = cv2.waitKey(1) & 0xFF
        if len(clicked_points) == 4:
            break
        if k == 27:  # ESC
            break

    cv2.destroyWindow("Calibration")

    if len(clicked_points) < 4:
        print("❌ キャリブレーション中断/未完了")
        return None

    return np.float32(clicked_points)

cam_pts_new = run_calibration(cap)
if cam_pts_new is None:
    raise SystemExit("❌ キャリブレーション未完了")

cam_pts = cam_pts_new
M    = compute_M(cam_pts, shift_x)
Minv = np.linalg.inv(M)
print("✅ 初回キャリブ完了（M/Minv生成）")

# =========================================================
# 🎨 色学習関数（P1:ピンク, P2:緑）
# =========================================================
MOMENT_MIN = 3500
SMOOTH_N   = 5

# 初期プレースホルダ
lower_pink_main = upper_pink_main = None
lower_pink_reflect = upper_pink_reflect = None
lower_pink_white = upper_pink_white = None
lower_green = upper_green = None
pink_main_range = pink_reflect_range = pink_white_range = None
green_range = None
learned_pink_hsv = (0.0,0.0,0.0)
learned_green_hsv = (0.0,0.0,0.0)

def learn_colors():
    global lower_pink_main, upper_pink_main
    global lower_pink_reflect, upper_pink_reflect
    global lower_pink_white, upper_pink_white
    global lower_green, upper_green
    global pink_main_range, pink_reflect_range, pink_white_range
    global green_range
    global learned_pink_hsv, learned_green_hsv

    # --- P1 Pink ---
    print("\n🎨 P1 ピンクの帽子を中央に置いて Spaceキーで登録（ESCで中断）")
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        h, w = frame.shape[:2]
        cx, cy = w//2, h//2

        cv2.rectangle(frame,(cx-25,cy-25),(cx+25,cy+25),(0,255,255),2)
        cv2.putText(frame,"P1(Pink): Spaceで登録 / ESCで中断",(30,40),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
        cv2.imshow("Color Learning", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            return
        if k == 32:
            roi = frame[cy-25:cy+25, cx-25:cx+25]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            h_mean_p, s_mean_p, v_mean_p = np.mean(hsv.reshape(-1,3), axis=0)

            H_RANGE_MAIN = 15
            lower_pink_main = np.array([
                max(0,   h_mean_p-H_RANGE_MAIN),
                max(40,  s_mean_p-90),
                max(40,  v_mean_p-90)
            ], np.uint8)
            upper_pink_main = np.array([
                min(179, h_mean_p+H_RANGE_MAIN),
                min(255, s_mean_p+90),
                min(255, v_mean_p+90)
            ], np.uint8)

            H_RANGE_REFLECT = 30
            lower_pink_reflect = np.array([
                max(0,   h_mean_p-H_RANGE_REFLECT),
                20,
                max(80,  v_mean_p-40)
            ], np.uint8)
            upper_pink_reflect = np.array([
                min(179, h_mean_p+H_RANGE_REFLECT),
                255, 255
            ], np.uint8)

            lower_pink_white = np.array([
                max(0,   h_mean_p-H_RANGE_REFLECT),
                70,
                140
            ], np.uint8)
            upper_pink_white = np.array([
                min(179, h_mean_p+H_RANGE_REFLECT),
                200,
                255
            ], np.uint8)

            learned_pink_hsv = (float(h_mean_p), float(s_mean_p), float(v_mean_p))
            print("🎀 ピンク再学習完了！")
            break

    # --- P2 Green ---
    print("\n🎨 P2 緑の帽子を中央に置いて Spaceキーで登録（ESCで中断）")
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        h, w = frame.shape[:2]
        cx, cy = w//2, h//2

        cv2.rectangle(frame,(cx-25,cy-25),(cx+25,cy+25),(0,255,0),2)
        cv2.putText(frame,"P2(Green): Spaceで登録 / ESCで中断",(30,40),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
        cv2.imshow("Color Learning", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            return
        if k == 32:
            roi = frame[cy-25:cy+25, cx-25:cx+25]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            h_mean_g, s_mean_g, v_mean_g = np.mean(hsv.reshape(-1,3), axis=0)

            H_RANGE = 15
            lower_green = np.array([
                max(0,   h_mean_g-H_RANGE),
                max(30,  s_mean_g-90),
                max(40,  v_mean_g-90)
            ], np.uint8)
            upper_green = np.array([
                min(179, h_mean_g+H_RANGE),
                min(255, s_mean_g+90),
                min(255, v_mean_g+90)
            ], np.uint8)

            learned_green_hsv = (float(h_mean_g), float(s_mean_g), float(v_mean_g))
            print("🍀 緑再学習完了！")
            break

    cv2.destroyWindow("Color Learning")

    pink_main_range    = (lower_pink_main,    upper_pink_main)
    pink_reflect_range = (lower_pink_reflect, upper_pink_reflect)
    pink_white_range   = (lower_pink_white,   upper_pink_white)
    green_range        = (lower_green, upper_green)

    print("🔄 色検出パラメータ再構築完了！")

# 初回色学習
learn_colors()

# =========================================================
# 🔎 blob選択ロジック（色学習のHSVプロファイル付き）
# =========================================================
def pick_blob_with_profile(mask, hsv_img, learned_hsv, prev_pos,
                           area_min=MOMENT_MIN,
                           h_tol=18, s_tol=80, v_tol=80,
                           highlight_dist=80):
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None

    lh,ls,lv = learned_hsv
    candidates = []

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < area_min:
            continue

        mm = cv2.moments(cnt)
        if mm["m00"] == 0:
            continue
        cx = int(mm["m10"]/mm["m00"])
        cy = int(mm["m01"]/mm["m00"])

        x,y,w,h = cv2.boundingRect(cnt)
        x2 = min(x+w, hsv_img.shape[1])
        y2 = min(y+h, hsv_img.shape[0])
        roi_hsv  = hsv_img[y:y2, x:x2]
        roi_mask = mask[y:y2, x:x2]
        ys, xs = np.where(roi_mask > 0)
        if len(xs) == 0:
            continue
        pixels = roi_hsv[ys, xs]
        h_mean,s_mean,v_mean = np.mean(pixels, axis=0)

        dh = abs(h_mean - lh)
        ds = abs(s_mean - ls)
        dv = abs(v_mean - lv)

        if dh > h_tol:
            continue

        if prev_pos is not None:
            dist = math.hypot(cx - prev_pos[0], cy - prev_pos[1])
        else:
            dist = 0.0

        ok_sv = (ds <= s_tol and dv <= v_tol)
        if not ok_sv and prev_pos is not None and dist < highlight_dist:
            ok_sv = True
        if not ok_sv:
            continue

        candidates.append((dist, area, cnt, (cx,cy)))

    if not candidates:
        return None, None

    if prev_pos is not None:
        candidates.sort(key=lambda x: x[0])
    else:
        candidates.sort(key=lambda x: -x[1])

    _,_,best_cnt,best_center = candidates[0]
    return best_cnt, best_center

# =========================================================
# 🧾 カメラ画面にグリッド描画（デバッグ）
# =========================================================
def draw_grid_on_camera(cam_img, xs, top_y, bottom_y, selected_idx):
    out = cam_img.copy()
    for i, gx in enumerate(xs):
        pts_proj = np.float32([[gx, top_y],[gx, bottom_y]]).reshape(-1,1,2)
        pts_cam  = cv2.perspectiveTransform(pts_proj, Minv).reshape(-1,2)
        p1 = tuple(np.int32(pts_cam[0]))
        p2 = tuple(np.int32(pts_cam[1]))
        color = (0,255,255) if i==selected_idx or i==selected_idx+1 else (0,200,200)
        cv2.line(out, p1, p2, color, 2)

    for i in range(8):
        quad = np.float32([
            [xs[i],   top_y],
            [xs[i+1], top_y],
            [xs[i+1], bottom_y],
            [xs[i],   bottom_y]
        ]).reshape(-1,1,2)
        quad_cam = cv2.perspectiveTransform(quad, Minv).reshape(-1,2).astype(np.int32)
        overlay = out.copy()
        cv2.fillConvexPoly(overlay, quad_cam, (0,255,255) if i==selected_idx else (0,180,180))
        alpha = 0.10 if i==selected_idx else 0.05
        out = cv2.addWeighted(overlay, alpha, out, 1-alpha, 0)

        center_proj = np.float32([[
            (xs[i]+xs[i+1])*0.5, max(30, top_y+30)
        ]]).reshape(-1,1,2)
        center_cam  = cv2.perspectiveTransform(center_proj, Minv).reshape(-1,2)[0]
        cv2.putText(out, NOTE_LABELS[i],
                    (int(center_cam[0]-10), int(center_cam[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    return out

# =========================================================
# 🧩 RGBA ブレンド（汎用）
# =========================================================
def blend_rgba(dst, src, x, y, fade):
    if src is None:
        return
    h,w = dst.shape[:2]
    fh,fw = src.shape[:2]
    x1,y1 = int(x - fw//2), int(y - fh//2)
    x2,y2 = x1+fw, y1+fh
    if x2<=0 or y2<=0 or x1>=w or y1>=h:
        return
    rx1,ry1 = max(0,x1), max(0,y1)
    rx2,ry2 = min(w,x2), min(h,y2)
    sx1,sy1 = rx1-x1, ry1-y1
    sx2,sy2 = sx1+(rx2-rx1), sy1+(ry2-ry1)

    roi  = dst[ry1:ry2, rx1:rx2]
    crop = src[sy1:sy2, sx1:sx2]
    if crop.size == 0:
        return

    # ✅ RGBA(4ch) = alpha を使って合成
    if crop.shape[2] >= 4:
        alpha = (crop[:,:,3].astype(np.float32)/255.0) * float(fade)
        alpha = alpha[...,None]
        roi[:] = ((1.0-alpha)*roi + alpha*crop[:,:,:3]).astype(np.uint8)
        return

    # ✅ RGB(3ch) = 上書きしないで “ふわっと” 合成（見えるように）
    cv2.addWeighted(crop, float(fade), roi, 1.0-float(fade), 0, roi)

# =========================================================
# 🧠 “オブジェクト一般” クラス（中身不問）
# =========================================================
class MovingObject:
    def __init__(self, img_left, img_right, is_rare=False):
        self.is_rare = is_rare

        base_img = img_left
        scale = np.random.uniform(0.25, 2.0)
        h, w = base_img.shape[:2]
        self.img_left  = cv2.resize(img_left,  (int(w*scale), int(h*scale)), cv2.INTER_AREA)
        self.img_right = cv2.resize(img_right, (int(w*scale), int(h*scale)), cv2.INTER_AREA)

        self.img = self.img_left
        self.h, self.w = self.img.shape[:2]

        self.alpha_max = np.random.uniform(0.4, 0.9)
        self.alpha = 0.0
        self.fade_in_speed = np.random.uniform(0.02, 0.05)
        self.fade_out_speed = np.random.uniform(0.015, 0.035)

        self.y = np.random.randint(-self.h, proj_h)

        speed = np.random.uniform(2.0, 10.0)
        self.x = 0
        self.vx = speed

        # “泳ぐ”ではなく “浮遊”にも使える揺れ
        self.t = 0.0
        self.amp = np.random.uniform(5, 20)
        self.freq = np.random.uniform(0.010, 0.025)

        self.birth = time.time()
        self.life_span = np.random.uniform(4.0, 7.0)
        self.dead = False
        self.caught = False

    def update(self):
        if self.caught:
            self.dead = True

        if not self.dead:
            self.alpha += self.fade_in_speed
            if self.alpha >= self.alpha_max:
                self.alpha = self.alpha_max
        else:
            self.alpha -= self.fade_out_speed
            if self.alpha <= 0:
                return False

        self.x += self.vx
        self.t += 1.0
        self.y_disp = self.y + np.sin(self.t * self.freq) * self.amp

        if not self.caught and (time.time() - self.birth) > self.life_span:
            self.dead = True

        if self.x < -self.w*2 or self.x > proj_w + self.w*2:
            return False

        return True

    def draw(self, frame):
        if self.alpha <= 0:
            return

        x1 = int(self.x)
        y1 = int(self.y_disp)
        x2 = x1 + self.w
        y2 = y1 + self.h

        if x2 <= 0 or y2 <= 0 or x1 >= proj_w or y1 >= proj_h:
            return

        rx1, ry1 = max(0, x1), max(0, y1)
        rx2, ry2 = min(proj_w, x2), min(proj_h, y2)
        sx1, sy1 = rx1 - x1, ry1 - y1
        sx2, sy2 = sx1 + (rx2 - rx1), sy1 + (ry2 - ry1)

        if sx2 <= sx1 or sy2 <= sy1:
            return

        roi = frame[ry1:ry2, rx1:rx2]
        crop = self.img[sy1:sy2, sx1:sx2]
        if crop.size == 0:
            return

        if crop.shape[2] == 4:
            rgb = crop[:, :, :3].astype(np.float32)
            alpha_c = (crop[:, :, 3:4].astype(np.float32) / 255.0) * self.alpha
            roi_f = roi.astype(np.float32)
            if alpha_c.shape[:2] != rgb.shape[:2]:
                return
            blended = roi_f * (1.0 - alpha_c) + rgb * alpha_c
            roi[:] = blended.astype(np.uint8)
        else:
            cv2.addWeighted(crop, self.alpha, roi, 1 - self.alpha, 0, roi)

def check_object_collision(obj, head_x, head_y):
    if head_x is None or head_y is None:
        return False
    x1 = obj.x
    y1 = obj.y_disp
    x2 = obj.x + obj.w
    y2 = obj.y_disp + obj.h
    margin = 10
    return (x1 - margin) <= head_x <= (x2 + margin) and (y1 - margin) <= head_y <= (y2 + margin)

# =========================================================
# ベース画像初期化
# =========================================================
base, xs_expand, xs_canon, top_y, bottom_y = build_base()

# =========================================================
# 🎬 無人デモモード管理
# =========================================================
DEMO_MODE = False
last_human_time = time.time()
IDLE_THRESHOLD = 25.0
DEMO_MAX_OBJECTS = 6
DEMO_RIPPLE_INTERVAL = (2.0, 5.0)
next_demo_ripple_time = time.time() + random.uniform(*DEMO_RIPPLE_INTERVAL)

# =========================================================
# 🔁 メインループ準備
# =========================================================
smooth1 = deque(maxlen=SMOOTH_N)
smooth2 = deque(maxlen=SMOOTH_N)

objects = []           # ← “fish” ではなく汎用 object
MAX_OBJECTS = 3

ripples = []
effect_flashes = []
EFFECT_SCALE = 0.20  # ← ここを変えれば全部の大きさが変わる


TRIGGER_COOLDOWN = 0.28
MISS_LIMIT = 10
last_note_time = {n:0 for n in NOTE_ORDER}
last_cell_1 = None
last_cell_2 = None
miss1 = 0
miss2 = 0
last_note_name = "-"
prev_t = time.time()

prev_p1_pos = None
prev_p2_pos = None

SONG_COOLDOWN = 1.0
last_song_time = 0
SONG_Y_TOLERANCE = 250
SHOW_GUIDE_RING = True
SHOW_PIANO_MARKER = True

# =========================================================
# 🔁 メインループ
# =========================================================
next_spawn_time = time.time() + 1.0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    now_main = time.time()
    dt_keys = now_main - prev_t
    update_key_lanes(dt_keys)

    if DEMO_MODE:
        for i in range(8):
            key_opacity[i] = KEY_TARGET_IDLE + 0.05 * math.sin(time.time()*0.8 + i)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    proj_display = base.copy()
    cam_vis = frame.copy()

    # ---------------------------
    # 🎨 P1: ピンク検出
    # ---------------------------
    low_main, up_main   = pink_main_range
    low_ref,  up_ref    = pink_reflect_range
    low_white, up_white = pink_white_range

    mask_main  = cv2.inRange(hsv, low_main,   up_main)
    mask_ref   = cv2.inRange(hsv, low_ref,    up_ref)
    mask_white = cv2.inRange(hsv, low_white,  up_white)

    mask_p = cv2.bitwise_or(mask_main, mask_ref)
    mask_p = cv2.bitwise_or(mask_p, mask_white)
    mask_p = cv2.erode(mask_p, None, 1)
    mask_p = cv2.dilate(mask_p, None, 1)
    mask_p = cv2.GaussianBlur(mask_p, (7,7), 0)

    cnt1, center1 = pick_blob_with_profile(mask_p, hsv, learned_pink_hsv, prev_p1_pos, area_min=MOMENT_MIN)

    detected1 = False
    new_cell_1 = None
    p1_px = p1_py = None

    if cnt1 is not None and center1 is not None:
        miss1 = 0
        cx,cy = center1
        prev_p1_pos = (cx,cy)
        smooth1.append((cx,cy))
        sx = int(np.mean([pp[0] for pp in smooth1]))
        sy = int(np.mean([pp[1] for pp in smooth1]))

        pt = cv2.perspectiveTransform(np.array([[[sx,sy]]],dtype=np.float32), M)[0][0]
        px, py = int(pt[0]), int(pt[1])
        if 0 <= py < proj_h:
            px = apply_height_correction(px, py)

        for i in range(8):
            if xs_expand[i] <= px <= xs_expand[i+1]:
                new_cell_1 = i
                break
        if new_cell_1 is None:
            new_cell_1 = 0 if px < xs_expand[0] else 7

        p1_px, p1_py = px, int(np.clip(py, top_y+20, bottom_y-20))
        detected1 = True

        cv2.drawContours(cam_vis, [cnt1], -1, (0,0,255), 2)
        cv2.circle(cam_vis, (sx,sy), 14, (0,0,255), 2)
        cv2.putText(cam_vis, f"P1({sx},{sy})", (sx+12, sy-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        if SHOW_PIANO_MARKER:
            center_x = int((xs_canon[new_cell_1] + xs_canon[new_cell_1+1]) / 2)
            cv2.drawMarker(proj_display, (center_x, p1_py), (0,120,255),
                           markerType=cv2.MARKER_TILTED_CROSS, markerSize=18, thickness=2)
    else:
        miss1 += 1
        if miss1 > MISS_LIMIT:
            smooth1.clear()
            prev_p1_pos = None
            last_cell_1 = None

    # ---------------------------
    # 🎨 P2: 緑検出
    # ---------------------------
    low_g, up_g = green_range
    mask_g = cv2.inRange(hsv, low_g, up_g)
    mask_g = cv2.erode(mask_g, None, 1)
    mask_g = cv2.dilate(mask_g, None, 1)
    mask_g = cv2.GaussianBlur(mask_g, (7,7), 0)

    cnt2, center2 = pick_blob_with_profile(mask_g, hsv, learned_green_hsv, prev_p2_pos, area_min=MOMENT_MIN)

    detected2 = False
    new_cell_2 = None
    p2_px = p2_py = None

    if cnt2 is not None and center2 is not None:
        miss2 = 0
        cx2,cy2 = center2
        prev_p2_pos = (cx2,cy2)
        smooth2.append((cx2,cy2))
        sx2 = int(np.mean([pp[0] for pp in smooth2]))
        sy2 = int(np.mean([pp[1] for pp in smooth2]))

        pt2 = cv2.perspectiveTransform(np.array([[[sx2,sy2]]],dtype=np.float32), M)[0][0]
        px2, py2 = int(pt2[0]), int(pt2[1])
        if 0 <= py2 < proj_h:
            px2 = apply_height_correction(px2, py2)

        for i in range(8):
            if xs_expand[i] <= px2 <= xs_expand[i+1]:
                new_cell_2 = i
                break
        if new_cell_2 is None:
            new_cell_2 = 0 if px2 < xs_expand[0] else 7

        p2_px, p2_py = px2, int(np.clip(py2, top_y+20, bottom_y-20))
        detected2 = True

        cv2.drawContours(cam_vis, [cnt2], -1, (0,255,0), 2)
        cv2.circle(cam_vis, (sx2,sy2), 14, (0,255,0), 2)
        cv2.putText(cam_vis, f"P2({sx2},{sy2})", (sx2+12, sy2-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        if SHOW_PIANO_MARKER:
            center_x2 = int((xs_canon[new_cell_2] + xs_canon[new_cell_2+1]) / 2)
            cv2.drawMarker(proj_display, (center_x2, p2_py), (0,255,150),
                           markerType=cv2.MARKER_TILTED_CROSS, markerSize=18, thickness=2)
    else:
        miss2 += 1
        if miss2 > MISS_LIMIT:
            smooth2.clear()
            prev_p2_pos = None
            last_cell_2 = None

    # =========================================================
    # 🧍 デモ遷移管理
    # =========================================================
    if detected1 or detected2:
        if DEMO_MODE:
            print("🌟 デモ解除（人を検出）")
        DEMO_MODE = False
        last_human_time = time.time()
    else:
        if not DEMO_MODE and (time.time() - last_human_time > IDLE_THRESHOLD):
            DEMO_MODE = True
            print("🌙 無人デモモードへ移行")

    # ---------------------------
    # 🔔 ノートトリガー
    # ---------------------------
    now = time.time()

    if MODE == 0:
        if detected1 and new_cell_1 is not None and new_cell_1 != last_cell_1:
            note = NOTE_ORDER[new_cell_1]
            if now - last_note_time[note] > TRIGGER_COOLDOWN:
                last_note_time[note] = now
                last_note_name = NOTE_LABELS[new_cell_1]
                if note in SOUNDS:
                    SOUNDS[note].play()
                    if p1_py is not None:
                        center_x = int((xs_canon[new_cell_1] + xs_canon[new_cell_1+1]) / 2)
                        t0 = time.time()
                        ripples.append(Ripple(center_x, p1_py, t0))
                        effect_flashes.append(EffectFlash(center_x, p1_py, t0))
                        activate_key_lane(new_cell_1)

            last_cell_1 = new_cell_1

        if detected2 and new_cell_2 is not None and new_cell_2 != last_cell_2:
            note2 = NOTE_ORDER[new_cell_2]
            if now - last_note_time[note2] > TRIGGER_COOLDOWN:
                last_note_time[note2] = now
                last_note_name = NOTE_LABELS[new_cell_2]
                if note2 in SOUNDS:
                    SOUNDS[note2].play()
                    if p2_py is not None:
                        center_x2 = int((xs_canon[new_cell_2] + xs_canon[new_cell_2+1]) / 2)
                        t0 = time.time()
                        ripples.append(Ripple(center_x2, p2_py, t0))
                        effect_flashes.append(EffectFlash(center_x2, p2_py, t0))
                        activate_key_lane(new_cell_2)

            last_cell_2 = new_cell_2

    elif MODE == 1:
        current_notes = SONGS[current_song_id]["notes"]
        if len(current_notes) > 0:
            current_note = current_notes[song_index]
            try:
                target_cell = NOTE_ORDER.index(current_note)
            except ValueError:
                target_cell = 0

            xs_canon2 = grid_boundaries_canonical(0.0)
            guide_x = int((xs_canon2[target_cell] + xs_canon2[target_cell+1]) / 2)
            guide_y = int((top_y + bottom_y) / 2)

            hit1 = detected1 and (new_cell_1 == target_cell) and (p1_py is not None) and (abs(p1_py - guide_y) < SONG_Y_TOLERANCE)
            hit2 = detected2 and (new_cell_2 == target_cell) and (p2_py is not None) and (abs(p2_py - guide_y) < SONG_Y_TOLERANCE)

            if now - last_song_time >= SONG_COOLDOWN and (hit1 or hit2):
                if current_note in SOUNDS:
                    SOUNDS[current_note].play()
                activate_key_lane(target_cell)
                effect_flashes.append(EffectFlash(guide_x, guide_y, time.time()))
                last_song_time = now
                song_index = (song_index + 1) % len(current_notes)
            
            if SHOW_GUIDE_RING:
                cv2.circle(proj_display, (guide_x, guide_y), 26, (0,200,255), 2)

    # =========================================================
    # 🧩 オブジェクトスポーン（デモ/通常で上限や間隔を変える）
    # =========================================================
    if DEMO_MODE:
        MAX_OBJECTS = DEMO_MAX_OBJECTS
        base_interval = 0.5
    else:
        MAX_OBJECTS = 3
        base_interval = 1.0

    now_spawn = time.time()
    if now_spawn >= next_spawn_time:
        if normal_left is None or normal_right is None:
            next_spawn_time = now_spawn + 2.0
        else:
            if len(objects) < MAX_OBJECTS:
                r = np.random.rand()
                is_rare = (r < 0.10) and (rare_left is not None) and (rare_right is not None)

                if is_rare:
                    obj = MovingObject(rare_left, rare_right, is_rare=True)
                else:
                    obj = MovingObject(normal_left, normal_right, is_rare=False)

                # 鍵盤画像の帯に寄せて出す（key がない場合は中央帯）
                if key_img is not None:
                    kh, kw = key_img.shape[:2]
                    key_y1 = int((proj_h - kh) / 2)
                    lane_min = key_y1 + int(kh * 0.05)
                    lane_max = key_y1 + int(kh * 0.95)
                    lane_max = min(lane_max, proj_h - obj.h - 10)
                    lane_min = max(0, lane_min)
                    if lane_max < lane_min:
                        lane_max = lane_min
                    obj.y = random.randint(lane_min, lane_max)
                else:
                    lane_min = int((top_y + bottom_y) / 2) - 120
                    lane_max = int((top_y + bottom_y) / 2) - 20
                    lane_min = max(0, lane_min)
                    lane_max = min(proj_h - obj.h - 10, max(lane_min, lane_max))
                    obj.y = random.randint(lane_min, lane_max)

                center_start = (np.random.rand() < 0.45)
                if not center_start:
                    dir_right = bool(np.random.rand() < 0.5)
                    if dir_right:
                        obj.img = obj.img_right
                        obj.x = -obj.w
                        obj.vx = abs(obj.vx)
                    else:
                        obj.img = obj.img_left
                        obj.x = proj_w + obj.w
                        obj.vx = -abs(obj.vx)
                else:
                    center_x = proj_w * 0.5
                    spread = proj_w * 0.12
                    obj.x = center_x + np.random.uniform(-spread, spread)
                    dir_right = bool(np.random.rand() < 0.5)
                    if dir_right:
                        obj.img = obj.img_right
                        obj.vx = abs(obj.vx)
                    else:
                        obj.img = obj.img_left
                        obj.vx = -abs(obj.vx)

                objects.append(obj)

            next_spawn_time = now_spawn + np.random.uniform(base_interval, base_interval + 2.0)

    # ① 鍵盤レーン
    proj_display = draw_keys(proj_display)

    # ② オブジェクト描画 + 当たり判定
    for obj in list(objects):
        alive = obj.update()

        if not obj.caught:
            hit = False
            if detected1 and p1_px is not None and p1_py is not None:
                if check_object_collision(obj, p1_px, p1_py):
                    hit = True
            if detected2 and p2_px is not None and p2_py is not None:
                if check_object_collision(obj, p2_px, p2_py):
                    hit = True

            if hit:
                if obj.is_rare:
                    if RARE_HIT_SFX: RARE_HIT_SFX.play()
                else:
                    if HIT_SFX: HIT_SFX.play()

                obj.caught = True
                obj.dead = True
                obj.fade_out_speed *= 3.0

                # hit_effect は EffectFlash として出す（サイズ統一）
                hx = int(obj.x + obj.w * 0.5)
                hy = int(obj.y_disp + obj.h * 0.5)
                effect_flashes.append(EffectFlash(hx, hy, time.time(), scale=EFFECT_SCALE))

        obj.draw(proj_display)

        if not alive:
            objects.remove(obj)

    # ③ Ripple描画
    now_r = time.time()
    alive_ripples = []
    for rp in ripples:
        if rp.is_alive(now_r):
            rp.draw(proj_display, now_r)
            alive_ripples.append(rp)
    ripples = alive_ripples

    # ④ EffectFlash描画（Rippleと同時に光らせる）
    now_fx = time.time()
    alive_fx = []
    for fx in effect_flashes:
        if fx.is_alive(now_fx):
            fx.draw(proj_display, now_fx, hit_effect)
            alive_fx.append(fx)
    effect_flashes = alive_fx


    # デモ中：ランダムRipple
    if DEMO_MODE:
        now_rp = time.time()
        if now_rp >= next_demo_ripple_time:
            rx = random.randint(100, proj_w - 100)
            ry = random.randint(top_y+50, bottom_y-50)
            ripples.append(Ripple(rx, ry, now_rp))
            next_demo_ripple_time = now_rp + random.uniform(*DEMO_RIPPLE_INTERVAL)

    # HUD
    now2 = time.time()
    fps = 1.0 / max(1e-6, (now2 - prev_t))
    prev_t = now2
    total_w = sum(abs_widths) * grid_scale_x
    approx_cell_w = (sum(abs_widths)/8.0) * grid_scale_x
    est_shift = height_corr_ratio * 0.5 * approx_cell_w * 5.0
    mode_name = MODE_NAMES[MODE]

    if MODE == 1:
        current_notes = SONGS[current_song_id]["notes"]
        song_name = SONGS[current_song_id]["name"]
        if len(current_notes) > 0:
            cn = current_notes[song_index]
            try:
                ci = NOTE_ORDER.index(cn)
                sl = NOTE_LABELS[ci]
            except ValueError:
                sl = "?"
            song_info = f"{song_name} {sl}({song_index+1}/{len(current_notes)})"
        else:
            song_info = f"{song_name} -"
    else:
        song_info = "-"

    hud = (
        f"FPS:{fps:.1f} "
        f"| MODE:{mode_name} "
        f"| DEMO:{'ON' if DEMO_MODE else 'OFF'} "
        f"| sel:{NOTE_LABELS[selected_idx]} "
        f"| TotalW:{total_w:.1f}px "
        f"| Last:{last_note_name} "
        f"| Song:{song_info} "
        f"| shift_x:{shift_x} "
        f"| off:{grid_offset_x:.1f} "
        f"| scale:{grid_scale_x:.3f} "
        f"| hgt:{grid_height_scale:.2f} "
        f"| Hcorr:{'ON' if height_corr_enabled else 'OFF'}"
        f"({height_corr_ratio:+.2f},~{est_shift:+.1f}px)"
    )

    cv2.imshow("Projector Output", proj_display)
    base_tmp, xs_tmp, _, top_tmp, bottom_tmp = build_base()
    cam_grid = draw_grid_on_camera(cam_vis, xs_tmp, top_tmp, bottom_tmp, selected_idx)
    ch, cw = cam_grid.shape[:2]
    cv2.putText(cam_grid, hud, (10, ch-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)
    cv2.imshow("Camera", cam_grid)

    # =========================================================
    # ⌨ キー入力
    # =========================================================
    k = cv2.waitKey(1) & 0xFF
    if k in (27, ord('q')):
        break

    elif k == ord('a'):
        shift_x -= 5
        M = compute_M(cam_pts, shift_x); Minv = np.linalg.inv(M)
    elif k == ord('d'):
        shift_x += 5
        M = compute_M(cam_pts, shift_x); Minv = np.linalg.inv(M)

    elif k == ord('j'):
        grid_offset_x -= 5
        base, xs_expand, xs_canon, top_y, bottom_y = build_base()
    elif k == ord('k'):
        grid_offset_x += 5
        base, xs_expand, xs_canon, top_y, bottom_y = build_base()
    elif k == ord('u'):
        grid_scale_x = max(0.50, grid_scale_x - 0.02)
        base, xs_expand, xs_canon, top_y, bottom_y = build_base()
    elif k == ord('o'):
        grid_scale_x = min(2.50, grid_scale_x + 0.02)
        base, xs_expand, xs_canon, top_y, bottom_y = build_base()

    elif k == ord('z'):
        grid_height_scale = max(0.30, grid_height_scale - 0.10)
        base, xs_expand, xs_canon, top_y, bottom_y = build_base()
    elif k == ord('x'):
        grid_height_scale = min(2.00, grid_height_scale + 0.10)
        base, xs_expand, xs_canon, top_y, bottom_y = build_base()

    elif MODE == 1 and k in (ord('1'), ord('2'), ord('3')):
        if k == ord('1'):
            current_song_id = "twinkle"
        elif k == ord('2'):
            current_song_id = "frog"
        elif k == ord('3'):
            current_song_id = "jingle"
        song_index = 0
        last_song_time = 0
        print(f"🎵 Song切替: {SONGS[current_song_id]['name']}")

    elif MODE == 0 and k in (
        ord('1'),ord('2'),ord('3'),ord('4'),
        ord('5'),ord('6'),ord('7'),ord('8')
    ):
        selected_idx = int(chr(k)) - 1
    elif k == ord('i'):
        selected_idx = (selected_idx - 1) % 8
    elif k == ord('l'):
        selected_idx = (selected_idx + 1) % 8

    elif k == ord('h'):
        height_corr_enabled = not height_corr_enabled
    elif k == ord('n'):
        height_corr_ratio = max(-0.40, height_corr_ratio - 0.05)
        base, xs_expand, xs_canon, top_y, bottom_y = build_base()
    elif k == ord('m'):
        height_corr_ratio = min( 0.40, height_corr_ratio + 0.05)
        base, xs_expand, xs_canon, top_y, bottom_y = build_base()

    elif k == ord('s'):
        save_named_settings()
    elif k == ord('r'):
        load_named_settings()
        base, xs_expand, xs_canon, top_y, bottom_y = build_base()
        print("🔄 設定を再読込しました。")

    elif k == ord('w'):
        print("🔄 鍵盤をすべてリセット（背景のみへ）")
        for i in range(8):
            key_opacity[i] = 0.0
            key_state[i] = 0

    elif k == ord('f'):
        save_named_settings("grid_room")
    elif k == ord('g'):
        save_named_settings("grid_outdoor")
    elif k == ord('p'):
        save_named_settings("grid_gym")

    elif k == ord('v'):
        load_named_settings("grid_room")
        base, xs_expand, xs_canon, top_y, bottom_y = build_base()
    elif k == ord('b'):
        load_named_settings("grid_outdoor")
        base, xs_expand, xs_canon, top_y, bottom_y = build_base()
    elif k == ord('c'):
        load_named_settings("grid_gym")
        base, xs_expand, xs_canon, top_y, bottom_y = build_base()

    elif k == ord('G'):
        show_proj_grid = not show_proj_grid
        base, xs_expand, xs_canon, top_y, bottom_y = build_base()
        print(f"🧱 Grid lines on Projector: {show_proj_grid}")
    elif k == ord('T'):
        show_proj_labels = not show_proj_labels
        base, xs_expand, xs_canon, top_y, bottom_y = build_base()
        print(f"🔤 Note labels on Projector: {show_proj_labels}")

    elif k == ord('e'):
        MODE = (MODE + 1) % 2
        print(f"🔁 MODE 切り替え: {MODE_NAMES[MODE]}")
        if MODE == 1:
            song_index = 0
            last_note_name = "-"

    elif k == ord('0'):
        song_index = 0
        last_song_time = 0
        print("🔁 Song Reset (manual)")

    elif k == ord('y'):
        SHOW_GUIDE_RING = not SHOW_GUIDE_RING
        print(f"🟡 Guide Ring: {'ON' if SHOW_GUIDE_RING else 'OFF'}")

    elif k == ord('Y'):
        SHOW_PIANO_MARKER = not SHOW_PIANO_MARKER
        print(f"🎹 Piano Marker: {'ON' if SHOW_PIANO_MARKER else 'OFF'}")

    elif k == ord('R'):
        print("\n🔁 色検出リセット → 再学習開始")
        learn_colors()
        smooth1.clear()
        smooth2.clear()
        prev_p1_pos = None
        prev_p2_pos = None
        miss1 = miss2 = 0
        last_cell_1 = last_cell_2 = None
        DEMO_MODE = False
        last_human_time = time.time()

    elif k == ord('C'):
        print("\n🧼 4点キャリブをリセット → 再キャリブ開始")
        cam_pts_new = run_calibration(cap)
        if cam_pts_new is not None:
            cam_pts = cam_pts_new
            M = compute_M(cam_pts, shift_x)
            Minv = np.linalg.inv(M)
            print("✅ キャリブ更新完了（M/Minv 更新）")

            smooth1.clear()
            smooth2.clear()
            prev_p1_pos = None
            prev_p2_pos = None
            miss1 = miss2 = 0
            last_cell_1 = last_cell_2 = None
        else:
            print("⚠ 再キャリブを中断しました（変更なし）")

cap.release()
cv2.destroyAllWindows()
