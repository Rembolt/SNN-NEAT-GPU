#!/usr/bin/env python3
r"""
snnd_to_video.py  –  converts a .snnd spike-frame file into an MP4
 
Usage:
    python "D:\Code\C++\SNN-GPU\debug\BinaryToVideo.py" "D:\Code\C++\SNN-GPU\frame-captures\spike_frames_v2.snnd" "D:\Code\C++\SNN-GPU\snn_frame_visualization_120grid.mp4" 12 1 4
 
Defaults: fps=12, block=1, scale=4
 
Requires: pip install numpy pillow "imageio[ffmpeg]"
"""
 
import sys
import struct
import numpy as np
from PIL import Image
import imageio.v2 as imageio
 
FILE_HEADER_SIZE = 28  # magic/u32 version/u32 width/u32 height/u32 features/u32 t0/u64
FRAME_PREFIX_SIZE = 16  # timestamp_us/u64 + frame_delta_us/u64

def feature_bytes_to_rgb(frame_features: np.ndarray) -> np.ndarray:
    """
    Map v2 one-byte-per-feature neurons to an RGB debug visualization:
      feature0 red, feature1 green, feature2 blue, feature3 yellow,
      feature4 white, feature5 black, feature6 movement, feature7 detail.
    """
    h, w, features = frame_features.shape
    planes = np.zeros((h, w, 8), dtype=np.uint8)
    planes[..., :min(features, 8)] = (frame_features[..., :min(features, 8)] != 0).astype(np.uint8)
    b0 = planes[..., 0]
    b1 = planes[..., 1]
    b2 = planes[..., 2]
    b3 = planes[..., 3]
    b4 = planes[..., 4]
    b5 = planes[..., 5]
    b6 = planes[..., 6]
    b7 = planes[..., 7]

    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Base color channels from semantic threshold bits.
    rgb[..., 0] = (b0 * 255 + b3 * 255)
    rgb[..., 1] = (b1 * 255 + b3 * 255)
    rgb[..., 2] = (b2 * 255)

    # White/black override luma-like state.
    rgb[b4 == 1] = np.array([255, 255, 255], dtype=np.uint8)
    rgb[b5 == 1] = np.array([0, 0, 0], dtype=np.uint8)

    # Overlay movement/detail to make temporal + texture spikes obvious.
    movement_mask = b6 == 1
    detail_mask = b7 == 1
    rgb[movement_mask] = np.maximum(rgb[movement_mask], np.array([255, 0, 255], dtype=np.uint8))
    rgb[detail_mask] = np.maximum(rgb[detail_mask], np.array([0, 255, 255], dtype=np.uint8))
    return rgb

def unpack_bitfield_to_rgb(frame_bits: np.ndarray) -> np.ndarray:
    """Map legacy v1 packed feature bits to the same RGB visualization."""
    feature_planes = np.stack(
        [((frame_bits >> bit) & 1).astype(np.uint8) for bit in range(8)],
        axis=-1,
    )
    return feature_bytes_to_rgb(feature_planes)
 
def pixelate(img: np.ndarray, block: int) -> np.ndarray:
    """Average each block×block tile → chunky pixel look."""
    if block <= 1:
        return img
    h, w = img.shape[:2]
    out = img.copy()
    for y in range(0, h, block):
        for x in range(0, w, block):
            tile = img[y:y+block, x:x+block]
            out[y:y+block, x:x+block] = tile.mean(axis=(0, 1), keepdims=True).astype(np.uint8)
    return out
 
def snnd_to_video(input_path, output_path, fps=12, block=4, scale=4):
    with open(input_path, "rb") as f:
        raw = f.read()
 
    magic = raw[0:4]
    assert magic == b"DNNS", f"Bad magic: {magic}"
    _, version, width, height, features, t0 = struct.unpack_from("<5IQ", raw, 0)
    if version not in (1, 2):
        raise RuntimeError(f"Unsupported snnd version: {version}")
 
    if version == 1:
        frame_size = width * height
    else:
        frame_size = width * height * features
    record_size = FRAME_PREFIX_SIZE + frame_size
    payload = len(raw) - FILE_HEADER_SIZE
    n_frames = payload // record_size
 
    block = max(1, int(block))
    scale = max(1, int(scale))
    effective_w = (width + block - 1) // block
    effective_h = (height + block - 1) // block

    print(f"[snnd] v{version}  {width}x{height}  features={features}  t0={t0}  frames={n_frames}  "
          f"fps={fps}  block={block}  scale={scale}")
    print(f"[snnd] effective visual grid: {effective_w}x{effective_h}")
    if payload % record_size:
        print(f"[snnd] warning: ignoring {payload % record_size} trailing bytes")
    if features != 8:
        print(f"[snnd] warning: expected 8 visual features, got {features}")
 
    out_w, out_h = width * scale, height * scale
 
    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
    )
 
    try:
        for i in range(n_frames):
            frame_offset = FILE_HEADER_SIZE + i * record_size
            timestamp_us, frame_delta_us = struct.unpack_from("<QQ", raw, frame_offset)
            frame = np.frombuffer(
                raw, dtype=np.uint8, offset=frame_offset + FRAME_PREFIX_SIZE, count=frame_size
            )
            if version == 1:
                rgb = unpack_bitfield_to_rgb(frame.reshape(height, width))
            else:
                rgb = feature_bytes_to_rgb(frame.reshape(height, width, features))
            rgb = pixelate(rgb, block)                         # chunky pixels
            img = Image.fromarray(rgb).resize(                 # scale up (nearest)
                (out_w, out_h), Image.NEAREST)
            writer.append_data(np.asarray(img))
            print(
                f"[snnd] frame {i+1}/{n_frames}  ts={timestamp_us}  dt_us={frame_delta_us}",
                flush=True,
            )
    finally:
        # Ensure moov atom and metadata are finalized even on interruption.
        writer.close()
 
    print(f"\n[snnd] done -> {output_path}")
 
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.snnd> <output.mp4> [fps] [block] [scale]")
        sys.exit(1)
    snnd_to_video(
        sys.argv[1], sys.argv[2],
        fps   = int(sys.argv[3]) if len(sys.argv) > 3 else 12,
        block = int(sys.argv[4]) if len(sys.argv) > 4 else 1,
        scale = int(sys.argv[5]) if len(sys.argv) > 5 else 4,
    )