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
    """Calculate optimal dimensions for ASCII rendering."""
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
# Color conversion & quantization
# -------------------------
def rgb_to_ansi256(r, g, b):
    """Convert RGB to closest ANSI 256 color code."""
    # Check grayscale (232-255: 24 grayscale colors)
    if r == g == b:
        if r < 8:
            return 16
        if r > 248:
            return 231
        return int(((r - 8) / 247) * 24) + 232
    
    # 16-231: 6x6x6 color cube
    r_idx = int(r / 255 * 5)
    g_idx = int(g / 255 * 5)
    b_idx = int(b / 255 * 5)
    return 16 + 36 * r_idx + 6 * g_idx + b_idx

def rgb_to_ansi_truecolor(r, g, b):
    """Return ANSI true color escape sequence for RGB."""
    return f"\x1b[38;2;{r};{g};{b}m"

def rgb_to_ansi_bg_truecolor(r, g, b):
    """Return ANSI true color escape sequence for RGB background."""
    return f"\x1b[48;2;{r};{g};{b}m"

# -------------------------
# Advanced image processing
# -------------------------
def apply_gamma(img, gamma=1.0):
    """Apply gamma correction with LUT."""
    if abs(gamma - 1.0) < 0.01:
        return img
    
    inv_gamma = 1.0 / gamma
    lut = np.clip((np.arange(256) / 255.0) ** inv_gamma * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)

def adjust_contrast(img, alpha=1.0, beta=0.0):
    """Adjust contrast and brightness."""
    if abs(alpha - 1.0) < 0.01 and abs(beta) < 0.01:
        return img
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def adjust_saturation(img_bgr, saturation=1.0):
    """Adjust color saturation."""
    if abs(saturation - 1.0) < 0.01:
        return img_bgr
    
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def unsharp_mask(img, sigma=1.0, strength=0.5):
    """Apply unsharp masking for enhanced detail."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    return sharpened

def bilateral_denoise(img, d=5, sigma_color=50, sigma_space=50):
    """Edge-preserving noise reduction."""
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

def adaptive_histogram_equalization(img_bgr, clip_limit=2.0):
    """Apply CLAHE in LAB color space for better color preservation."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# -------------------------
# Dithering algorithms
# -------------------------
def floyd_steinberg_color_dither(img_rgb, palette_colors):
    """Floyd-Steinberg dithering for color images."""
    h, w, _ = img_rgb.shape
    buf = img_rgb.astype(np.float32).copy()
    out = np.zeros_like(img_rgb, dtype=np.uint8)
    
    for y in range(h):
        for x in range(w):
            old_color = buf[y, x]
            
            # Find nearest palette color
            distances = np.sum((palette_colors - old_color) ** 2, axis=1)
            nearest_idx = np.argmin(distances)
            new_color = palette_colors[nearest_idx]
            
            out[y, x] = new_color
            err = old_color - new_color
            
            # Distribute error
            if x + 1 < w:
                buf[y, x + 1] += err * 0.4375
            if y + 1 < h:
                if x > 0:
                    buf[y + 1, x - 1] += err * 0.1875
                buf[y + 1, x] += err * 0.3125
                if x + 1 < w:
                    buf[y + 1, x + 1] += err * 0.0625
    
    return out

def simple_color_quantize(img_rgb, levels=6):
    """Simple color quantization to reduce palette."""
    step = 256 // levels
    quantized = (img_rgb // step) * step + step // 2
    return np.clip(quantized, 0, 255).astype(np.uint8)

# -------------------------
# High-quality color rendering
# -------------------------
def frame_to_ansi_color_halfblocks(frame_bgr, cols=160, gamma=1.0, alpha=1.1, beta=0,
                                   saturation=1.2, enhance=True, sharpen=0.3, 
                                   denoise=False, use_clahe=False, 
                                   color_mode='truecolor', dither_colors=False):
    """
    Convert frame to colored ASCII with half-block characters.
    
    Args:
        color_mode: 'truecolor' (24-bit) or '256' (ANSI 256 colors)
        dither_colors: Apply dithering to colors (for 256-color mode)
    """
    h, w = frame_bgr.shape[:2]
    cols = max(20, cols)
    w_px, h_px, _ = target_dims(h, w, cols, char_aspect=0.5, use_half_blocks=True)
    
    # High-quality resize using Lanczos
    resized = cv2.resize(frame_bgr, (w_px, h_px), interpolation=cv2.INTER_LANCZOS4)
    
    # Optional denoising
    if denoise:
        resized = bilateral_denoise(resized)
    
    # Optional CLAHE
    if use_clahe:
        resized = adaptive_histogram_equalization(resized, clip_limit=2.5)
    
    # Tone adjustments
    resized = apply_gamma(resized, gamma=gamma)
    resized = adjust_contrast(resized, alpha=alpha, beta=beta)
    
    # Saturation adjustment
    resized = adjust_saturation(resized, saturation=saturation)
    
    # Sharpening
    if enhance and sharpen > 0:
        resized = unsharp_mask(resized, sigma=0.8, strength=sharpen)
    
    # Convert to RGB for processing
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Optional color dithering for 256-color mode
    if color_mode == '256' and dither_colors:
        # Create ANSI 256-color palette
        palette = []
        for r in range(6):
            for g in range(6):
                for b in range(6):
                    palette.append([r * 51, g * 51, b * 51])
        palette = np.array(palette, dtype=np.float32)
        rgb = floyd_steinberg_color_dither(rgb, palette)
    elif color_mode == '256':
        # Simple quantization
        rgb = simple_color_quantize(rgb, levels=6)
    
    # Convert to grayscale for luminance-based character selection
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    
    # Build output with colors
    lines = []
    reset = "\x1b[0m"
    
    for y in range(0, h_px, 2):
        line_parts = []
        
        for x in range(w_px):
            # Get colors for top and bottom pixels
            top_rgb = rgb[y, x]
            bot_rgb = rgb[y + 1, x] if y + 1 < h_px else top_rgb
            
            # Get luminance for character selection
            top_lum = gray[y, x]
            bot_lum = gray[y + 1, x] if y + 1 < h_px else top_lum
            
            # Determine character based on luminance
            avg_lum = (int(top_lum) + int(bot_lum)) / 2
            
            # Choose character and coloring strategy
            if abs(int(top_lum) - int(bot_lum)) < 30:
                # Similar luminance - use full block with average color
                avg_rgb = ((top_rgb.astype(np.int32) + bot_rgb.astype(np.int32)) // 2).astype(np.uint8)
                char = '█'
                
                if color_mode == 'truecolor':
                    color_code = rgb_to_ansi_truecolor(avg_rgb[0], avg_rgb[1], avg_rgb[2])
                else:
                    ansi_code = rgb_to_ansi256(avg_rgb[0], avg_rgb[1], avg_rgb[2])
                    color_code = f"\x1b[38;5;{ansi_code}m"
                
                line_parts.append(f"{color_code}{char}")
            else:
                # Different luminance - use half blocks with foreground/background colors
                if top_lum > bot_lum:
                    char = '▀'
                    fg_rgb = top_rgb
                    bg_rgb = bot_rgb
                else:
                    char = '▄'
                    fg_rgb = bot_rgb
                    bg_rgb = top_rgb
                
                if color_mode == 'truecolor':
                    fg_code = rgb_to_ansi_truecolor(fg_rgb[0], fg_rgb[1], fg_rgb[2])
                    bg_code = rgb_to_ansi_bg_truecolor(bg_rgb[0], bg_rgb[1], bg_rgb[2])
                    line_parts.append(f"{fg_code}{bg_code}{char}")
                else:
                    fg_ansi = rgb_to_ansi256(fg_rgb[0], fg_rgb[1], fg_rgb[2])
                    bg_ansi = rgb_to_ansi256(bg_rgb[0], bg_rgb[1], bg_rgb[2])
                    line_parts.append(f"\x1b[38;5;{fg_ansi}m\x1b[48;5;{bg_ansi}m{char}")
        
        lines.append(''.join(line_parts) + reset)
    
    return "\n".join(lines)

# -------------------------
# Frame timer
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

# -------------------------
# Player
# -------------------------
def supports_ansi():
    """Check if terminal supports ANSI escape codes."""
    return sys.stdout.isatty()

def fast_clear_and_home():
    """Move cursor to home and clear screen."""
    sys.stdout.write("\x1b[H\x1b[2J")

def play(video_path, cols=160, gamma=1.0, alpha=1.1, beta=0, saturation=1.2,
         max_fps=None, show_stats=False, enhance=True, sharpen=0.3, 
         denoise=False, use_clahe=False, color_mode='truecolor', dither_colors=False):
    """
    Play video as high-quality colored ASCII art.
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
    print(f"🎨 Color: {color_mode.upper()} | Saturation: {saturation}x")
    print(f"🔧 Tone: γ={gamma} | α={alpha} | β={beta}")
    
    features = []
    if enhance:
        features.append(f"sharpen={sharpen}")
    if denoise:
        features.append("denoise")
    if use_clahe:
        features.append("CLAHE")
    if dither_colors:
        features.append("color-dither")
    if features:
        print(f"✨ Enhancements: {', '.join(features)}")
    
    print("\n⏳ Starting in 2 seconds... (Press Ctrl+C to stop)")
    time.sleep(2)
    
    # Initialize
    use_ansi = supports_ansi()
    if not use_ansi:
        print("⚠️  Warning: Terminal may not support ANSI colors properly")
    
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
            
            frame_txt = frame_to_ansi_color_halfblocks(
                frame, cols=cols, gamma=gamma, alpha=alpha, beta=beta,
                saturation=saturation, enhance=enhance, sharpen=sharpen,
                denoise=denoise, use_clahe=use_clahe, 
                color_mode=color_mode, dither_colors=dither_colors
            )
            
            if use_ansi:
                fast_clear_and_home()
                sys.stdout.write(frame_txt)
                
                if show_stats:
                    fps = timer.get_fps()
                    progress = (frame_num / total_frames * 100) if total_frames > 0 else 0
                    elapsed = time.perf_counter() - start_time
                    sys.stdout.write(f"\x1b[0m\n\n📊 Frame: {frame_num}/{total_frames} ({progress:.1f}%) | "
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
            sys.stdout.write("\x1b[0m")  # Reset colors
            sys.stdout.write("\x1b[?25h")  # Show cursor
            sys.stdout.flush()
        cap.release()
        
        elapsed = time.perf_counter() - start_time
        avg_fps = frame_num / elapsed if elapsed > 0 else 0
        print(f"\n🎬 Finished! Played {frame_num} frames in {elapsed:.1f}s (avg {avg_fps:.1f} FPS)")
        if timer.dropped_frames > 0:
            print(f"⚠️  Dropped {timer.dropped_frames} frames")

# -------------------------
# CLI with color presets
# -------------------------
if __name__ == "__main__":
    import argparse
    import glob
    
    # Quality presets for color mode
    PRESETS = {
        'draft': {
            'cols': 100, 'color_mode': '256', 'enhance': False,
            'sharpen': 0, 'gamma': 1.0, 'alpha': 1.1, 'beta': 0, 
            'saturation': 1.0, 'dither_colors': False
        },
        'balanced': {
            'cols': 140, 'color_mode': 'truecolor', 'enhance': True,
            'sharpen': 0.2, 'gamma': 1.0, 'alpha': 1.1, 'beta': 0,
            'saturation': 1.2, 'dither_colors': False
        },
        'quality': {
            'cols': 160, 'color_mode': 'truecolor', 'enhance': True,
            'sharpen': 0.3, 'gamma': 1.0, 'alpha': 1.15, 'beta': 0,
            'saturation': 1.3, 'dither_colors': False
        },
        'ultra': {
            'cols': 200, 'color_mode': 'truecolor', 'enhance': True,
            'sharpen': 0.4, 'gamma': 1.0, 'alpha': 1.2, 'beta': 0,
            'saturation': 1.4, 'use_clahe': True, 'dither_colors': False
        },
        'vivid': {
            'cols': 160, 'color_mode': 'truecolor', 'enhance': True,
            'sharpen': 0.5, 'gamma': 0.95, 'alpha': 1.3, 'beta': 5,
            'saturation': 1.8, 'use_clahe': True, 'dither_colors': False
        }
    }
    
    parser = argparse.ArgumentParser(
        description="High-Quality COLOR ASCII Video Player",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Quality Presets:
  draft     - Fast preview (100 cols, 256 colors)
  balanced  - Good balance (140 cols, true color) [DEFAULT]
  quality   - High quality (160 cols, enhanced colors)
  ultra     - Maximum quality (200 cols, CLAHE)
  vivid     - Extra vibrant colors (high saturation)

Examples:
  %(prog)s video.mp4 --preset quality --stats
  %(prog)s video.mp4 --preset vivid
  %(prog)s video.mp4 --cols 180 --saturation 1.5 --sharpen 0.4
  %(prog)s video.mp4 --preset ultra --denoise --fps 24
  %(prog)s video.mp4 --color-mode 256  # For older terminals
        """
    )
    
    parser.add_argument("video", nargs='?', help="Path to video file")
    parser.add_argument("--preset", choices=PRESETS.keys(), default='balanced',
                       help="Quality preset (default: balanced)")
    parser.add_argument("--cols", type=int, help="Number of columns")
    parser.add_argument("--gamma", type=float, help="Gamma correction")
    parser.add_argument("--alpha", type=float, help="Contrast multiplier")
    parser.add_argument("--beta", type=float, help="Brightness offset")
    parser.add_argument("--saturation", type=float, help="Color saturation (1.0 = normal)")
    parser.add_argument("--color-mode", choices=['truecolor', '256'], 
                       help="Color mode: truecolor (24-bit) or 256 (8-bit)")
    parser.add_argument("--fps", type=float, help="Cap framerate")
    parser.add_argument("--sharpen", type=float, help="Sharpening strength (0.0-1.0)")
    parser.add_argument("--no-enhance", action="store_true", help="Disable enhancements")
    parser.add_argument("--denoise", action="store_true", help="Apply denoising")
    parser.add_argument("--use-clahe", action="store_true", help="Use adaptive histogram equalization")
    parser.add_argument("--dither-colors", action="store_true", help="Apply color dithering")
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
    
    # Apply preset, then override with explicit arguments
    config = PRESETS[args.preset].copy()
    
    if args.cols is not None:
        config['cols'] = args.cols
    if args.gamma is not None:
        config['gamma'] = args.gamma
    if args.alpha is not None:
        config['alpha'] = args.alpha
    if args.beta is not None:
        config['beta'] = args.beta
    if args.saturation is not None:
        config['saturation'] = args.saturation
    if args.color_mode is not None:
        config['color_mode'] = args.color_mode
    if args.sharpen is not None:
        config['sharpen'] = args.sharpen
    if args.no_enhance:
        config['enhance'] = False
    if args.denoise:
        config['denoise'] = True
    if args.use_clahe:
        config['use_clahe'] = True
    if args.dither_colors:
        config['dither_colors'] = True
    
    play(args.video, 
         cols=config['cols'],
         gamma=config.get('gamma', 1.0),
         alpha=config.get('alpha', 1.1),
         beta=config.get('beta', 0),
         saturation=config.get('saturation', 1.2),
         max_fps=args.fps,
         show_stats=args.stats,
         enhance=config.get('enhance', True),
         sharpen=config.get('sharpen', 0.3),
         denoise=config.get('denoise', False),
         use_clahe=config.get('use_clahe', False),
         color_mode=config.get('color_mode', 'truecolor'),
         dither_colors=config.get('dither_colors', False))