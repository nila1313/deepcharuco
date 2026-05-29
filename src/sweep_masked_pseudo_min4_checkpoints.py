from pathlib import Path
import subprocess
import re
import csv
import math


def patch_file(path, replacements):
    p = Path(path)
    txt = p.read_text()

    for pattern, repl in replacements:
        txt = re.sub(pattern, repl, txt)

    p.write_text(txt)


def summarize_csv(csv_path):
    rows = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    opencv = [int(r["opencv_corners"]) for r in rows]
    deep = [int(r["deepcharuco_corners_unique"]) for r in rows]
    matched = [int(r["matched_ids"]) for r in rows]

    errors = []
    duplicate_frames = 0

    for r in rows:
        if r["mean_error_px"] != "":
            errors.append(float(r["mean_error_px"]))

        if r["duplicate_deep_ids"] not in ["[]", "", None]:
            duplicate_frames += 1

    errors_sorted = sorted(errors)

    if errors_sorted:
        mid = len(errors_sorted) // 2
        if len(errors_sorted) % 2 == 1:
            median_error = errors_sorted[mid]
        else:
            median_error = 0.5 * (errors_sorted[mid - 1] + errors_sorted[mid])
    else:
        median_error = math.nan

    return {
        "opencv_frames": sum(x > 0 for x in opencv),
        "deep_frames": sum(x > 0 for x in deep),
        "matched_frames": sum(x > 0 for x in matched),
        "max_deep_corners": max(deep) if deep else 0,
        "avg_deep_corners": sum(deep) / len(deep) if deep else 0,
        "duplicate_frames": duplicate_frames,
        "mean_error_px": sum(errors) / len(errors) if errors else math.nan,
        "median_error_px": median_error,
    }


ckpt_dir = Path("./tb_logs/ckpts_deepcharuco_masked_pseudo_min4")

ckpts = sorted(
    ckpt_dir.glob("*.ckpt"),
    key=lambda p: p.stat().st_mtime
)

if len(ckpts) == 0:
    raise RuntimeError(f"No checkpoints found in {ckpt_dir}")

print("Found checkpoints:")
for c in ckpts:
    print(" ", c)

all_results = []

for ckpt in ckpts:
    safe_name = ckpt.stem.replace("=", "_").replace("-", "_")

    infer_img_dir = f"../my_dataset/inference_output_masked_pseudo_min4_sweep/{safe_name}/images"
    infer_kpt_dir = f"../my_dataset/inference_output_masked_pseudo_min4_sweep/{safe_name}/keypoints"
    comp_dir = f"../my_dataset/comparison_output_masked_pseudo_min4_sweep/{safe_name}"

    print("\n==============================")
    print("Evaluating:", ckpt)
    print("==============================")

    patch_file(
        "run_custom_inference.py",
        [
            (r'image_dir = ".*?"', 'image_dir = "../my_dataset/raw_frames"'),
            (r'output_img_dir = ".*?"', f'output_img_dir = "{infer_img_dir}"'),
            (r'output_kpt_dir = ".*?"', f'output_kpt_dir = "{infer_kpt_dir}"'),
            (r'deepc_path = ".*?"', f'deepc_path = "{ckpt}"'),
        ],
    )

    subprocess.run(["python", "run_custom_inference.py"], check=True)

    patch_file(
        "compare_deepcharuco_opencv.py",
        [
            (r'image_dir = ".*?"', 'image_dir = "../my_dataset/raw_frames"'),
            (r'deep_keypoint_dir = ".*?"', f'deep_keypoint_dir = "{infer_kpt_dir}"'),
            (r'output_dir = ".*?"', f'output_dir = "{comp_dir}"'),
        ],
    )

    subprocess.run(["python", "compare_deepcharuco_opencv.py"], check=True)

    summary_path = Path(comp_dir) / "comparison_summary.csv"

    metrics = summarize_csv(summary_path)
    metrics["checkpoint"] = str(ckpt)
    metrics["summary_path"] = str(summary_path)

    all_results.append(metrics)


out_path = Path("../my_dataset/masked_pseudo_min4_checkpoint_sweep_summary.csv")

fieldnames = [
    "checkpoint",
    "opencv_frames",
    "deep_frames",
    "matched_frames",
    "max_deep_corners",
    "avg_deep_corners",
    "duplicate_frames",
    "mean_error_px",
    "median_error_px",
    "summary_path",
]

with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_results)

print("\nSweep finished.")
print("Saved:", out_path)
print("\nResults:")

for r in all_results:
    print(
        Path(r["checkpoint"]).name,
        "deep_frames=", r["deep_frames"],
        "matched_frames=", r["matched_frames"],
        "avg_deep=", round(r["avg_deep_corners"], 3),
        "mean_error=", r["mean_error_px"],
        "median_error=", r["median_error_px"],
    )