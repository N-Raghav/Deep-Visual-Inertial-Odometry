"""
Render camera frames in Blender along a precomputed trajectory.
Usage: blender --background --python blender_render.py -- --data ../data/run_001 --textures ../data/textures
"""

import argparse
import csv
import json
import math
import os
import random
import sys

import numpy as np
import bpy


def parse_argv():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--textures", required=True)
    ap.add_argument("--plane-size", type=float, default=120.0)
    ap.add_argument("--tex-scale", type=float, default=1.0)
    ap.add_argument("--tex-rot", type=float, default=None)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def read_poses_cam(data_dir):
    data = np.loadtxt(os.path.join(data_dir, "poses_cam.csv"), delimiter=",", skiprows=1)
    # columns: t, x, y, z, qw, qx, qy, qz
    return data


def list_textures(tex_dir):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = [os.path.join(tex_dir, f) for f in sorted(os.listdir(tex_dir))
             if f.lower().endswith(exts)]
    if not files:
        raise RuntimeError(f"no textures found in {tex_dir}")
    return files


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.textures,
                  bpy.data.images, bpy.data.cameras, bpy.data.lights):
        for d in list(block):
            block.remove(d)


def make_plane(size, tex_path, tex_scale, tex_rot):
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    plane = bpy.context.active_object

    mat = bpy.data.materials.new("FloorMat")
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)

    out      = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf     = nt.nodes.new("ShaderNodeBsdfDiffuse")
    img_node = nt.nodes.new("ShaderNodeTexImage")
    mapping  = nt.nodes.new("ShaderNodeMapping")
    texcoord = nt.nodes.new("ShaderNodeTexCoord")

    img_node.image = bpy.data.images.load(tex_path)
    mapping.inputs["Scale"].default_value    = (tex_scale, tex_scale, 1.0)
    mapping.inputs["Rotation"].default_value = (0.0, 0.0, tex_rot)

    nt.links.new(texcoord.outputs["UV"],      mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"],   img_node.inputs["Vector"])
    nt.links.new(img_node.outputs["Color"],   bsdf.inputs["Color"])
    nt.links.new(bsdf.outputs["BSDF"],        out.inputs["Surface"])

    plane.data.materials.append(mat)


def setup_camera(K, img_w, img_h):
    cam_data = bpy.data.cameras.new("Cam")
    cam_obj  = bpy.data.objects.new("Cam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    fx, fy = K[0][0], K[1][1]
    sensor_w_mm = 36.0
    cam_data.sensor_fit   = "HORIZONTAL"
    cam_data.sensor_width = sensor_w_mm
    cam_data.lens         = fx * sensor_w_mm / img_w  # f_mm = fx_px * sensor_w / img_w

    scene = bpy.context.scene
    scene.render.resolution_x      = img_w
    scene.render.resolution_y      = img_h
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x    = 1.0
    scene.render.pixel_aspect_y    = fx / fy
    return cam_obj


def add_light():
    light_data = bpy.data.lights.new("Sun", type="SUN")
    light_data.energy = 4.0
    light_obj = bpy.data.objects.new("Sun", light_data)
    light_obj.location = (0, 0, 20)
    bpy.context.collection.objects.link(light_obj)


def configure_renderer(samples):
    scene = bpy.context.scene
    for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            scene.render.engine = engine
            break
        except TypeError:
            continue
    eevee = getattr(scene, "eevee", None)
    if eevee is not None and hasattr(eevee, "taa_render_samples"):
        eevee.taa_render_samples = samples
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode  = "RGB"
    scene.view_settings.view_transform      = "Standard"
    print(f"renderer: {scene.render.engine}  gpu_backend: {bpy.context.preferences.system.gpu_backend}")


def main():
    args = parse_argv()
    random.seed(args.seed)

    data_dir = os.path.abspath(args.data)
    img_dir  = os.path.join(data_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)

    poses    = read_poses_cam(data_dir)
    stride   = int(meta["camera_stride"])
    K        = meta["K"]
    img_w    = int(meta["image_w"])
    img_h    = int(meta["image_h"])
    nominal_h = float(meta["camera_height_nominal"])

    # account for ~20 deg of body tilt from quadrotor pitch/roll
    fov_x = 2 * math.atan(img_w / (2 * K[0][0])) + math.radians(40)
    traj_p = meta.get("trajectory_params", {})
    traj_radius = max(traj_p.get("A", 0.0), traj_p.get("B", 0.0), 0.0)
    safe_size = 2 * (nominal_h * math.tan(fov_x / 2) + traj_radius) * 1.5
    if args.plane_size < safe_size:
        print(f"[warn] plane_size={args.plane_size} may show edges; recommended >= {safe_size:.1f}")

    clear_scene()

    tex_scale = args.tex_scale
    tex_rot   = args.tex_rot if args.tex_rot is not None else random.uniform(0.0, math.pi)
    tex_path  = random.choice(list_textures(os.path.abspath(args.textures)))
    print(f"texture={os.path.basename(tex_path)}  scale={tex_scale:.2f}  rot={tex_rot:.2f}")

    make_plane(args.plane_size, tex_path, tex_scale, tex_rot)
    cam = setup_camera(K, img_w, img_h)
    add_light()
    configure_renderer(args.samples)

    cam_indices = range(0, len(poses), stride)
    scene = bpy.context.scene
    rows_index = []

    for k, idx in enumerate(cam_indices):
        row = poses[idx]
        t, pos, quat = row[0], row[1:4], row[4:8]

        cam.location = pos
        cam.rotation_mode = "QUATERNION"
        cam.rotation_quaternion = quat

        out_path = os.path.join(img_dir, f"{k:06d}.png")
        scene.render.filepath = out_path
        bpy.ops.render.render(write_still=True)

        rows_index.append((t, idx, os.path.relpath(out_path, data_dir)))
        if k % 50 == 0:
            print(f"[{k+1}/{len(cam_indices)}]  t={t:.3f}s")

    with open(os.path.join(data_dir, "frame_index.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "imu_index", "image_path"])
        w.writerows(rows_index)

    render_meta = {
        "plane_size_m": args.plane_size,
        "texture": os.path.basename(tex_path),
        "tex_scale": tex_scale,
        "tex_rot": tex_rot,
        "K": K,
        "image_w": img_w,
        "image_h": img_h,
        "n_frames": len(rows_index),
        "engine": scene.render.engine,
        "samples": args.samples,
    }
    with open(os.path.join(data_dir, "render_meta.json"), "w") as f:
        json.dump(render_meta, f, indent=2)

    print(f"rendered {len(rows_index)} frames -> {img_dir}")


if __name__ == "__main__":
    main()
