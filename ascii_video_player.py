import os
import sys
import time
import shutil
import numpy as np
import cv2
from collections import deque

# -------------------------
# Terminal / sizing helpers
# -------------------------
def term_size(fallback=(120, 32)):
    """Get terminal dimensions with fallback."""
    try:
        sz = shutil.get_terminal_size(fallback=fallback)
        return sz.columns, sz.lines
    except Exception:
        return fallback

def target_dims(src_h, src_w, cols, char_aspect=0.5, use_half_blocks=True):
    """
    Calculate optimal dimensions for ASCII rendering.
    char_aspect: Adjusted to 0.5 for better proportions (was 0.55)
    """
    w_px = max(20, cols)
    _, term_rows = term_size()
    rows_text = max(10, term_rows - 3)
    
    if use_half_blocks:
        h_px = max(10, int((src_h / src_w) * w_px * char_aspect))
        h_px = (h_px + 1) & ~1  # Ensure even
        max_h_px = max(10, rows_text * 2)
        h_px = min(h_px, max_h_px)
        rows_text = h_px // 2
    else:
        h_px = max(10, int((src_h / src_w) * w_px * char_aspect))
        h_px = min(h_px, rows_text)
        rows_text = h_px
    
    return w_px, h_px, rows_text

# -------------------------
# Advanced image processing
# -------------------------
def apply_gamma(gray, gamma=1.0):
    """Apply gamma correction with LUT."""
    if abs(gamma - 1.0) < 0.01:
        return gray
    
    inv_gamma = 1.0 / gamma
    lut = np.clip((np.arange(256) / 255.0) ** inv_gamma * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(gray, lut)

def adjust_contrast(gray, alpha=1.0, beta=0.0):
    """Adjust contrast and brightness."""
    if abs(alpha - 1.0) < 0.01 and abs(beta) < 0.01:
        return gray
    return cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

def adaptive_histogram_equalization(gray, clip_limit=2.0, tile_size=8):
    """Apply CLAHE for better local contrast."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(gray)

def unsharp_mask(gray, sigma=1.0, strength=0.5):
    """Apply unsharp masking for enhanced edge detail."""
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    sharpened = cv2.addWeighted(gray, 1.0 + strength, blurred, -strength, 0)
    return sharpened

def bilateral_denoise(gray, d=5, sigma_color=50, sigma_space=50):
    """Edge-preserving noise reduction."""
    return cv2.bilateralFilter(gray, d, sigma_color, sigma_space)

def atkinson_dither(gray_u8):
    """
    Atkinson dithering - produces high-quality results with good detail preservation.
    Similar to Floyd-Steinberg but distributes less error (75% vs 100%).
    """
    h, w = gray_u8.shape
    buf = gray_u8.astype(np.float32)
    out = np.zeros((h, w), dtype=bool)
    
    for y in range(h):
        for x in range(w):
            old = buf[y, x]
            new = old >= 128
            out[y, x] = not new
            err = (old - (255 if new else 0)) * 0.125  # Distribute 1/8 of error
            
            # Atkinson pattern (distributes to 6 neighbors)
            if x + 1 < w:
                buf[y, x + 1] += err
            if x + 2 < w:
                buf[y, x + 2] += err
            if y + 1 < h:
                if x > 0:
                    buf[y + 1, x - 1] += err
                buf[y + 1, x] += err
                if x + 1 < w:
                    buf[y + 1, x + 1] += err
            if y + 2 < h:
                buf[y + 2, x] += err
    
    return out

def stucki_dither(gray_u8):
    """
    Stucki dithering - distributes error to more neighbors for smoother gradients.
    Best for high-resolution output.
    """
    h, w = gray_u8.shape
    buf = gray_u8.astype(np.float32)
    out = np.zeros((h, w), dtype=bool)
    
    # Stucki coefficients (sum = 42)
    coeffs = [
        [(0, 1, 8), (0, 2, 4)],
        [(-2, 1, 2), (-1, 1, 4), (0, 1, 8), (1, 1, 4), (2, 1, 2)],
        [(-2, 2, 1), (-1, 2, 2), (0, 2, 4), (1, 2, 2), (2, 2, 1)]
    ]
    
    for y in range(h):
        for x in range(w):
            old = buf[y, x]
            new = old >= 128
            out[y, x] = not new
            err = old - (255 if new else 0)
            
            for row_offsets in coeffs:
                for dx, dy, weight in row_offsets:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        buf[ny, nx] += err * (weight / 42.0)
    
    return out

def floyd_steinberg_dither(gray_u8):
    """Floyd-Steinberg dithering with serpentine scanning."""
    h, w = gray_u8.shape
    buf = gray_u8.astype(np.float32)
    out = np.zeros((h, w), dtype=bool)
    
    for y in range(h):
        serpentine = (y & 1)
        
        if not serpentine:
            for x in range(w):
                old = buf[y, x]
                new = old >= 128
                out[y, x] = not new
                err = old - (255 if new else 0)
                
                if x + 1 < w:
                    buf[y, x + 1] += err * 0.4375
                if y + 1 < h:
                    if x > 0:
                        buf[y + 1, x - 1] += err * 0.1875
                    buf[y + 1, x] += err * 0.3125
                    if x + 1 < w:
                        buf[y + 1, x + 1] += err * 0.0625
        else:
            for x in range(w - 1, -1, -1):
                old = buf[y, x]
                new = old >= 128
                out[y, x] = not new
                err = old - (255 if new else 0)
                
                if x > 0:
                    buf[y, x - 1] += err * 0.4375
                if y + 1 < h:
                    if x + 1 < w:
                        buf[y + 1, x + 1] += err * 0.1875
                    buf[y + 1, x] += err * 0.3125
                    if x > 0:
                        buf[y + 1, x - 1] += err * 0.0625
    
    return out

# 8x8 Bayer matrix for smoother gradients
_BAYER_8X8 = np.array([
    [ 0, 32,  8, 40,  2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44,  4, 36, 14, 46,  6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [ 3, 35, 11, 43,  1, 33,  9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47,  7, 39, 13, 45,  5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21]
], dtype=np.float32) * 4.0  # Scale to [0..255]

def ordered_dither(gray_u8, matrix_size=8):
    """High-quality ordered dithering with 8x8 Bayer matrix."""
    h, w = gray_u8.shape
    
    if matrix_size == 8:
        matrix = _BAYER_8X8
        tile_size = 8
    else:
        # Fallback to 4x4
        matrix = (np.array([
            [ 0,  8,  2, 10],
            [12,  4, 14,  6],
            [ 3, 11,  1,  9],
            [15,  7, 13,  5]
        ], dtype=np.float32) + 0.5) * 16.0
        tile_size = 4
    
    tile_h = (h + tile_size - 1) // tile_size
    tile_w = (w + tile_size - 1) // tile_size
    tiled = np.tile(matrix, (tile_h, tile_w))[:h, :w]
    
    return gray_u8.astype(np.float32) < tiled

# -------------------------
# High-quality rendering
# -------------------------
_CHAR_MAP = ['█', '▀', '▄', ' ']

def pair_to_char_vectorized(top_black, bot_black):
    """Vectorized character mapping."""
    indices = (~top_black).astype(np.uint8) * 2 + (~bot_black).astype(np.uint8)
    return indices

def frame_to_ascii_halfblocks(frame_bgr, cols=160, gamma=1.0, alpha=1.2, beta=-10,
                              dither="atkinson", enhance=True, sharpen=0.3, 
                              denoise=False, use_clahe=False):
    """
    High-quality frame to ASCII conversion.
    
    Args:
        enhance: Apply sharpening and advanced processing
        sharpen: Sharpening strength (0.0-1.0)
        denoise: Apply edge-preserving denoising
        use_clahe: Use adaptive histogram equalization
    """
    h, w = frame_bgr.shape[:2]
    cols = max(20, cols)
    w_px, h_px, _ = target_dims(h, w, cols, char_aspect=0.5, use_half_blocks=True)
    
    # High-quality resize using Lanczos interpolation
    resized = cv2.resize(frame_bgr, (w_px, h_px), interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Optional denoising (before other processing)
    if denoise:
        gray = bilateral_denoise(gray)
    
    # Optional CLAHE for better local contrast
    if use_clahe:
        gray = adaptive_histogram_equalization(gray, clip_limit=2.5, tile_size=8)
    
    # Tone adjustments
    gray = apply_gamma(gray, gamma=gamma)
    gray = adjust_contrast(gray, alpha=alpha, beta=beta)
    
    # Sharpening for better detail
    if enhance and sharpen > 0:
        gray = unsharp_mask(gray, sigma=0.8, strength=sharpen)
    
    # High-quality dithering
    if dither == "atkinson":
        blacks = atkinson_dither(gray)
    elif dither == "stucki":
        blacks = stucki_dither(gray)
    elif dither == "fs":
        blacks = floyd_steinberg_dither(gray)
    elif dither == "ordered":
        blacks = ordered_dither(gray, matrix_size=8)
    else:
        blacks = gray < 128
    
    # Vectorized character generation
    lines = []
    for y in range(0, h_px, 2):
        top = blacks[y]
        bottom = blacks[y + 1] if y + 1 < h_px else np.zeros_like(top, dtype=bool)
        
        indices = pair_to_char_vectorized(top, bottom)
        line = ''.join(_CHAR_MAP[i] for i in indices)
        lines.append(line)
    
    return "\n".join(lines)

# -------------------------
# Enhanced player
# -------------------------
class FrameTimer:
    """Accurate frame timing with drift correction."""
    
    def __init__(self, target_fps):
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps if target_fps > 0 else 0.0333
        self.next_frame_time = time.perf_counter()
        self.frame_count = 0
        self.fps_samples = deque(maxlen=30)
        self.last_fps_update = time.perf_counter()
        self.dropped_frames = 0
    
    def wait(self):
        """Wait until next frame should be displayed."""
        now = time.perf_counter()
        wait_time = self.next_frame_time - now
        
        if wait_time > 0:
            time.sleep(wait_time)
        elif wait_time < -self.frame_time:
            # We're falling behind - count dropped frames
            self.dropped_frames += 1
        
        self.next_frame_time = max(
            time.perf_counter(),
            self.next_frame_time + self.frame_time
        )
        
        self.frame_count += 1
        
        now = time.perf_counter()
        if now - self.last_fps_update >= 1.0:
            self.fps_samples.append(self.frame_count)
            self.frame_count = 0
            self.last_fps_update = now
    
    def get_fps(self):
        """Get current FPS measurement."""
        return sum(self.fps_samples) / len(self.fps_samples) if self.fps_samples else 0

def supports_ansi():
    """Check if terminal supports ANSI escape codes."""
    return sys.stdout.isatty()

def fast_clear_and_home():
    """Move cursor to home and clear screen."""
    sys.stdout.write("\x1b[H\x1b[2J")

def play(video_path, cols=160, gamma=1.0, alpha=1.2, beta=-10, 
         dither="atkinson", max_fps=None, show_stats=False,
         enhance=True, sharpen=0.3, denoise=False, use_clahe=False):
    """
    Play video as high-quality ASCII art.
    """
    if not os.path.exists(video_path):
        print(f"❌ File not found: {video_path}")
        sys.exit(1)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open video.")
        sys.exit(1)
    
    # Get video properties
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / src_fps if src_fps > 0 else 0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    target_fps = max_fps if (max_fps and max_fps > 0) else (src_fps if src_fps > 0 else 30)
    
    print(f"🎬 Playing: {os.path.basename(video_path)}")
    print(f"📊 Source: {src_w}x{src_h} @ {int(src_fps)} FPS, {total_frames} frames, {duration:.1f}s")
    print(f"⚙️  Render: {cols} cols @ {target_fps:.1f} FPS")
    print(f"🎨 Dither: {dither} | γ={gamma} | α={alpha} | β={beta}")
    
    features = []
    if enhance:
        features.append(f"sharpen={sharpen}")
    if denoise:
        features.append("denoise")
    if use_clahe:
        features.append("CLAHE")
    if features:
        print(f"✨ Enhancements: {', '.join(features)}")
    
    print("\n⏳ Starting in 2 seconds... (Press Ctrl+C to stop)")
    time.sleep(2)
    
    # Initialize
    use_ansi = supports_ansi()
    if use_ansi:
        sys.stdout.write("\x1b[?25l")  # Hide cursor
        sys.stdout.write("\x1b[2J")     # Clear screen
        sys.stdout.flush()
    
    timer = FrameTimer(target_fps)
    frame_num = 0
    start_time = time.perf_counter()
    
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            frame_txt = frame_to_ascii_halfblocks(
                frame, cols=cols, gamma=gamma, alpha=alpha, 
                beta=beta, dither=dither, enhance=enhance,
                sharpen=sharpen, denoise=denoise, use_clahe=use_clahe
            )
            
            if use_ansi:
                fast_clear_and_home()
                sys.stdout.write(frame_txt)
                
                if show_stats:
                    fps = timer.get_fps()
                    progress = (frame_num / total_frames * 100) if total_frames > 0 else 0
                    elapsed = time.perf_counter() - start_time
                    sys.stdout.write(f"\n\n📊 Frame: {frame_num}/{total_frames} ({progress:.1f}%) | "
                                   f"FPS: {fps:.1f} | Elapsed: {elapsed:.1f}s")
                    if timer.dropped_frames > 0:
                        sys.stdout.write(f" | Dropped: {timer.dropped_frames}")
                
                sys.stdout.flush()
            else:
                os.system("cls" if os.name == "nt" else "clear")
                print(frame_txt)
            
            timer.wait()
            frame_num += 1
    
    except KeyboardInterrupt:
        print("\n\n⏸️  Playback interrupted.")
    finally:
        if use_ansi:
            sys.stdout.write("\x1b[?25h")  # Show cursor
            sys.stdout.flush()
        cap.release()
        
        elapsed = time.perf_counter() - start_time
        avg_fps = frame_num / elapsed if elapsed > 0 else 0
        print(f"\n🎬 Finished! Played {frame_num} frames in {elapsed:.1f}s (avg {avg_fps:.1f} FPS)")
        if timer.dropped_frames > 0:
            print(f"⚠️  Dropped {timer.dropped_frames} frames")

# -------------------------
# CLI with presets
# -------------------------
if __name__ == "__main__":
    import argparse
    import glob
    
    # Quality presets
    PRESETS = {
        'draft': {
            'cols': 100, 'dither': 'ordered', 'enhance': False, 
            'sharpen': 0, 'gamma': 1.0, 'alpha': 1.15, 'beta': -5
        },
        'balanced': {
            'cols': 140, 'dither': 'fs', 'enhance': True, 
            'sharpen': 0.2, 'gamma': 1.0, 'alpha': 1.18, 'beta': -8
        },
        'quality': {
            'cols': 160, 'dither': 'atkinson', 'enhance': True, 
            'sharpen': 0.3, 'gamma': 1.0, 'alpha': 1.2, 'beta': -10
        },
        'ultra': {
            'cols': 200, 'dither': 'stucki', 'enhance': True, 
            'sharpen': 0.4, 'gamma': 1.0, 'alpha': 1.25, 'beta': -12,
            'use_clahe': True
        }
    }
    
    parser = argparse.ArgumentParser(
        description="High-Quality ASCII Video Player",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Quality Presets:
  draft     - Fast, lower quality (100 cols, ordered dither)
  balanced  - Good balance (140 cols, Floyd-Steinberg) [DEFAULT]
  quality   - High quality (160 cols, Atkinson dither, sharpening)
  ultra     - Maximum quality (200 cols, Stucki dither, CLAHE)

Examples:
  %(prog)s video.mp4 --preset quality
  %(prog)s video.mp4 --cols 180 --dither stucki --sharpen 0.5
  %(prog)s video.mp4 --preset ultra --fps 24 --stats
  %(prog)s video.mp4 --denoise --use-clahe
        """
    )
    
    parser.add_argument("video", nargs='?', help="Path to video file")
    parser.add_argument("--preset", choices=PRESETS.keys(), default='balanced',
                       help="Quality preset (default: balanced)")
    parser.add_argument("--cols", type=int, help="Number of columns")
    parser.add_argument("--gamma", type=float, help="Gamma correction")
    parser.add_argument("--alpha", type=float, help="Contrast multiplier")
    parser.add_argument("--beta", type=float, help="Brightness offset")
    parser.add_argument("--dither", choices=["atkinson", "stucki", "fs", "ordered", "threshold"],
                       help="Dithering algorithm")
    parser.add_argument("--fps", type=float, help="Cap framerate")
    parser.add_argument("--sharpen", type=float, help="Sharpening strength (0.0-1.0)")
    parser.add_argument("--no-enhance", action="store_true", help="Disable enhancements")
    parser.add_argument("--denoise", action="store_true", help="Apply denoising")
    parser.add_argument("--use-clahe", action="store_true", help="Use adaptive histogram equalization")
    parser.add_argument("--stats", action="store_true", help="Show performance statistics")
    
    args = parser.parse_args()
    
    # Handle video selection
    if not args.video:
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm', '*.flv', '*.wmv']
        videos = []
        for ext in video_extensions:
            videos.extend(glob.glob(ext))
        
        if videos:
            print("🎬 No video specified. Found these videos:")
            for i, v in enumerate(videos, 1):
                size = os.path.getsize(v) / (1024*1024)
                print(f"  {i}. {v} ({size:.1f} MB)")
            
            try:
                choice = input(f"\nSelect video (1-{len(videos)}) or Enter to exit: ").strip()
                if choice and choice.isdigit() and 1 <= int(choice) <= len(videos):
                    args.video = videos[int(choice) - 1]
                else:
                    sys.exit(0)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                sys.exit(0)
        else:
            print("❌ No video file specified and none found in directory.")
            print(f"\nUsage: python {os.path.basename(__file__)} <video_file> [options]")
            sys.exit(1)
    
    # Apply preset, then override with any explicit arguments
    config = PRESETS[args.preset].copy()
    
    if args.cols is not None:
        config['cols'] = args.cols
    if args.gamma is not None:
        config['gamma'] = args.gamma
    if args.alpha is not None:
        config['alpha'] = args.alpha
    if args.beta is not None:
        config['beta'] = args.beta
    if args.dither is not None:
        config['dither'] = args.dither
    if args.sharpen is not None:
        config['sharpen'] = args.sharpen
    if args.no_enhance:
        config['enhance'] = False
    if args.denoise:
        config['denoise'] = True
    if args.use_clahe:
        config['use_clahe'] = True
    
    play(args.video, 
         cols=config['cols'],
         gamma=config['gamma'],
         alpha=config['alpha'],
         beta=config['beta'],
         dither=config['dither'],
         max_fps=args.fps,
         show_stats=args.stats,
         enhance=config.get('enhance', True),
         sharpen=config.get('sharpen', 0.3),
         denoise=config.get('denoise', False),
         use_clahe=config.get('use_clahe', False))