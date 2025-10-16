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

def target_dims(src_h, src_w, cols, char_aspect=0.55, use_half_blocks=True):
    """
    Calculate optimal dimensions for ASCII rendering.
    
    Args:
        src_h, src_w: Source video dimensions
        cols: Target column width
        char_aspect: Character height/width ratio
        use_half_blocks: Use half-block characters (2 pixels per char)
    
    Returns:
        (w_px, h_px, rows_text): Width, height, and text rows
    """
    w_px = max(20, cols)
    _, term_rows = term_size()
    rows_text = max(10, term_rows - 3)  # Reserve 3 rows for UI
    
    if use_half_blocks:
        h_px = max(10, int((src_h / src_w) * w_px * char_aspect))
        h_px = (h_px + 1) & ~1  # Ensure even number (bit trick)
        max_h_px = max(10, rows_text * 2)
        h_px = min(h_px, max_h_px)
        rows_text = h_px // 2
    else:
        h_px = max(10, int((src_h / src_w) * w_px * char_aspect))
        h_px = min(h_px, rows_text)
        rows_text = h_px
    
    return w_px, h_px, rows_text

# -------------------------
# Image processing (optimized)
# -------------------------
def apply_gamma(gray, gamma=1.0):
    """Apply gamma correction with LUT for performance."""
    if abs(gamma - 1.0) < 0.01:
        return gray
    
    # Cache LUT generation (could be done once globally)
    inv_gamma = 1.0 / gamma
    lut = np.clip((np.arange(256) / 255.0) ** inv_gamma * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(gray, lut)

def adjust_contrast(gray, alpha=1.0, beta=0.0):
    """Adjust contrast and brightness efficiently."""
    if abs(alpha - 1.0) < 0.01 and abs(beta) < 0.01:
        return gray
    return cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

def floyd_steinberg_dither(gray_u8):
    """
    Floyd-Steinberg dithering with optimized serpentine scanning.
    Returns boolean mask: True=black, False=white.
    """
    h, w = gray_u8.shape
    buf = gray_u8.astype(np.float32)
    out = np.zeros((h, w), dtype=bool)
    
    for y in range(h):
        serpentine = (y & 1)  # Bit trick instead of modulo
        
        if not serpentine:
            for x in range(w):
                old = buf[y, x]
                new = old >= 128
                out[y, x] = not new  # Invert: True=black
                err = old - (255 if new else 0)
                
                # Distribute error
                if x + 1 < w:
                    buf[y, x + 1] += err * 0.4375  # 7/16
                if y + 1 < h:
                    if x > 0:
                        buf[y + 1, x - 1] += err * 0.1875  # 3/16
                    buf[y + 1, x] += err * 0.3125  # 5/16
                    if x + 1 < w:
                        buf[y + 1, x + 1] += err * 0.0625  # 1/16
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

# Optimized Bayer matrix (pre-computed)
_BAYER_4X4 = (np.array([
    [ 0,  8,  2, 10],
    [12,  4, 14,  6],
    [ 3, 11,  1,  9],
    [15,  7, 13,  5]
], dtype=np.float32) + 0.5) * 16.0  # Scale to [0..255]

def ordered_dither(gray_u8):
    """Fast ordered dithering using Bayer matrix."""
    h, w = gray_u8.shape
    
    # Tile the matrix efficiently
    tile_h, tile_w = (h + 3) // 4, (w + 3) // 4
    tiled = np.tile(_BAYER_4X4, (tile_h, tile_w))[:h, :w]
    
    return gray_u8.astype(np.float32) < tiled

# -------------------------
# Rendering (optimized)
# -------------------------
# Pre-computed character map for speed
_CHAR_MAP = ['█', '▀', '▄', ' ']

def pair_to_char_vectorized(top_black, bot_black):
    """Vectorized character mapping."""
    # Create index: 0=both black, 1=top only, 2=bottom only, 3=both white
    indices = (~top_black).astype(np.uint8) * 2 + (~bot_black).astype(np.uint8)
    return indices

def frame_to_ascii_halfblocks(frame_bgr, cols=120, gamma=1.0, alpha=1.15, beta=-8,
                              dither="fs", use_prefilter=True):
    """
    Convert frame to ASCII with half-block characters.
    
    Args:
        frame_bgr: Input frame in BGR format
        cols: Number of columns to render
        gamma: Gamma correction value
        alpha: Contrast multiplier
        beta: Brightness offset
        dither: Dithering method ("fs", "ordered", or "threshold")
        use_prefilter: Apply median blur to reduce noise
    
    Returns:
        String containing ASCII art
    """
    h, w = frame_bgr.shape[:2]
    cols = max(20, cols)
    w_px, h_px, _ = target_dims(h, w, cols, char_aspect=0.55, use_half_blocks=True)
    
    # Resize + grayscale (INTER_AREA is best for downsampling)
    resized = cv2.resize(frame_bgr, (w_px, h_px), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Tone adjustments
    gray = apply_gamma(gray, gamma=gamma)
    gray = adjust_contrast(gray, alpha=alpha, beta=beta)
    
    # Optional noise reduction
    if use_prefilter and min(w_px, h_px) > 80:
        gray = cv2.medianBlur(gray, 3)
    
    # Dithering
    if dither == "fs":
        blacks = floyd_steinberg_dither(gray)
    elif dither == "ordered":
        blacks = ordered_dither(gray)
    else:
        blacks = gray < 128
    
    # Vectorized character generation
    lines = []
    for y in range(0, h_px, 2):
        top = blacks[y]
        bottom = blacks[y + 1] if y + 1 < h_px else np.zeros_like(top, dtype=bool)
        
        # Vectorized mapping
        indices = pair_to_char_vectorized(top, bottom)
        line = ''.join(_CHAR_MAP[i] for i in indices)
        lines.append(line)
    
    return "\n".join(lines)

# -------------------------
# Player (enhanced)
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
    
    def wait(self):
        """Wait until next frame should be displayed."""
        now = time.perf_counter()
        wait_time = self.next_frame_time - now
        
        if wait_time > 0:
            time.sleep(wait_time)
        
        # Update timing for next frame (prevent drift)
        self.next_frame_time = max(
            time.perf_counter(),
            self.next_frame_time + self.frame_time
        )
        
        self.frame_count += 1
        
        # Calculate actual FPS
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
    return sys.stdout.isatty() and os.name != 'nt' or os.environ.get('TERM')

def fast_clear_and_home():
    """Move cursor to home and clear screen using ANSI codes."""
    sys.stdout.write("\x1b[H\x1b[2J")

def play(video_path, cols=140, gamma=1.0, alpha=1.18, beta=-8, 
         dither="fs", max_fps=None, show_stats=False):
    """
    Play video as ASCII art in terminal.
    
    Args:
        video_path: Path to video file
        cols: Number of columns to render
        gamma: Gamma correction
        alpha: Contrast multiplier
        beta: Brightness offset
        dither: Dithering method ("fs", "ordered", "threshold")
        max_fps: Cap framerate (None = use source FPS)
        show_stats: Display performance statistics
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
    
    target_fps = max_fps if (max_fps and max_fps > 0) else (src_fps if src_fps > 0 else 30)
    
    print(f"🎬 Playing: {os.path.basename(video_path)}")
    print(f"📊 {int(src_fps)} FPS, {total_frames} frames, {duration:.1f}s")
    print(f"⚙️  Rendering at {cols} cols, {target_fps:.1f} FPS")
    print(f"🎨 Dither: {dither}, γ={gamma}, α={alpha}, β={beta}")
    time.sleep(1)
    
    # Initialize
    use_ansi = supports_ansi()
    if use_ansi:
        sys.stdout.write("\x1b[?25l")  # Hide cursor
        sys.stdout.write("\x1b[2J")     # Clear screen
        sys.stdout.flush()
    
    timer = FrameTimer(target_fps)
    frame_num = 0
    
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            frame_txt = frame_to_ascii_halfblocks(
                frame, cols=cols, gamma=gamma, alpha=alpha, 
                beta=beta, dither=dither
            )
            
            if use_ansi:
                fast_clear_and_home()
                sys.stdout.write(frame_txt)
                
                if show_stats:
                    fps = timer.get_fps()
                    progress = (frame_num / total_frames * 100) if total_frames > 0 else 0
                    sys.stdout.write(f"\n\n📊 Frame: {frame_num}/{total_frames} | "
                                   f"{progress:.1f}% | {fps:.1f} FPS")
                
                sys.stdout.flush()
            else:
                os.system("cls" if os.name == "nt" else "clear")
                print(frame_txt)
            
            timer.wait()
            frame_num += 1
    
    except KeyboardInterrupt:
        pass
    finally:
        if use_ansi:
            sys.stdout.write("\x1b[?25h")  # Show cursor
            sys.stdout.flush()
        cap.release()
        print(f"\n\n🎬 Finished! Played {frame_num} frames.")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(
        description="ASCII Video Player - Convert videos to terminal art",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s video.mp4
  %(prog)s video.mp4 --cols 160 --fps 24
  %(prog)s video.mp4 --gamma 0.95 --alpha 1.22 --beta -6
  %(prog)s video.mp4 --dither ordered --stats
        """
    )
    
    parser.add_argument("video", nargs='?', help="Path to video file")
    parser.add_argument("--cols", type=int, default=140,
                       help="Number of columns (default: 140)")
    parser.add_argument("--gamma", type=float, default=1.0,
                       help="Gamma correction (default: 1.0)")
    parser.add_argument("--alpha", type=float, default=1.18,
                       help="Contrast multiplier (default: 1.18)")
    parser.add_argument("--beta", type=float, default=-8,
                       help="Brightness offset (default: -8)")
    parser.add_argument("--dither", choices=["fs", "ordered", "threshold"],
                       default="fs", help="Dithering method (default: fs)")
    parser.add_argument("--fps", type=float, default=None,
                       help="Cap framerate (default: source FPS)")
    parser.add_argument("--stats", action="store_true",
                       help="Show performance statistics")
    
    args = parser.parse_args()
    
    if not args.video:
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm', '*.flv', '*.wmv']
        videos = []
        for ext in video_extensions:
            videos.extend(glob.glob(ext))
        
        if videos:
            print("🎬 No video specified. Found these videos in current directory:")
            for i, v in enumerate(videos, 1):
                size = os.path.getsize(v) / (1024*1024)  # MB
                print(f"  {i}. {v} ({size:.1f} MB)")
            
            try:
                choice = input(f"\nSelect video (1-{len(videos)}) or press Enter to exit: ").strip()
                if choice and choice.isdigit() and 1 <= int(choice) <= len(videos):
                    args.video = videos[int(choice) - 1]
                else:
                    print("Exiting.")
                    sys.exit(0)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                sys.exit(0)
        else:
            print("❌ No video file specified and no videos found in current directory.")
            print("\nUsage:")
            print(f"  python {os.path.basename(__file__)} <video_file>")
            print(f"\nExample:")
            print(f"  python {os.path.basename(__file__)} my_video.mp4")
            print(f"\nSupported formats: MP4, AVI, MOV, MKV, WebM, FLV, WMV")
            sys.exit(1)
    
    play(args.video, cols=args.cols, gamma=args.gamma, alpha=args.alpha,
         beta=args.beta, dither=args.dither, max_fps=args.fps,
         show_stats=args.stats)