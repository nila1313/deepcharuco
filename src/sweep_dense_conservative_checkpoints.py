from pathlib import Path
import subprocess
import re
import csv
import math


def patch_file(path, replacements):
    p = Path(path)
    txt = p.read_text()

    for pattern, repl in replacements:
        txt = re.sub(pattern, repl, txt, flags=re.M)

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
    good_frames = 0
    borderline_frames = 0

    for r in rows:
        dup = str(r["duplicate_deep_ids"]).strip()
        has_dup = dup not in ["[]", "", "nan", "None"]

        if has_dup:
            duplicate_frames += 1

        err_text = str(r["mean_error_px"]).strip()

        if err_text not in ["", "nan", "None"]:
            err = float(err_text)
            errors.append(err)

            m = int(r["matched_ids"])

            if m >= 3 and err < 40 and not has_dup:
                good_frames += 1

            if m >= 3 and err < 70:
                borderline_frames += 1

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
        "total_frames": len(rows),
        "opencv_frames": sum(x > 0 for x in opencv),
        "deep_frames": sum(x > 0 for x in deep),
        "matched_frames": sum(x > 0 for x in matched),
        "max_deep_corners": max(deep) if deep else 0,
        "avg_deep_corners": sum(deep) / len(deep) if deep else 0,
        "duplicate_frames": duplicate_frames,
        "mean_error_px": sum(errors) / len(errors) if errors else math.nan,
        "median_error_px": median_error,
        "good_frames": good_frames,
        "borderline_frames": borderline_frames,
    }


ckpts = sorted(
    Path("tb_logs/ckpts_deepcharuco_dense_conservative").glob("*.ckpt"),
    key=lambda p: p.stat().st_mtime
)

results = []

for ckpt in ckpts:
    tag = ckpt.stem.replace("=", "").replace("-", "_")

    print()
    print("====================================")
    print("Evaluating:", ckpt)
    print("Tag:", tag)
    print("====================================")

    infer_img_dir = f"../my_dataset/inference_output_dense_conservative_sweep/{tag}/images"
    infer_kpt_dir = f"../my_dataset/inference_output_dense_conservative_sweep/{tag}/keypoints"
    comp_dir = f"../my_dataset/comparison_output_dense_conservative_sweep/{tag}"

    patch_file(
        "run_custom_inference.py",
        [
            (r'^output_img_dir = ".*"$', f'output_img_dir = "{infer_img_dir}"'),
            (r'^output_kpt_dir = ".*"$', f'output_kpt_dir = "{infer_kpt_dir}"'),
            (r'^deepc_path = ".*"$', f'deepc_path = "./{ckpt}"'),
            (r'^refinenet_path = .*$', 'refinenet_path = None'),
        ],
    )

    subprocess.run(["python", "run_custom_inference.py"], check=True)

    patch_file(
        "compare_deepcharuco_opencv.py",
        [
            (r'^deep_keypoint_dir = ".*"$', f'deep_keypoint_dir = "{infer_kpt_dir}"'),
            (r'^output_dir = ".*"$', f'output_dir = "{comp_dir}"'),
        ],
    )

    subprocess.run(["python", "compare_deepcharuco_opencv.py"], check=True)

    summary_path = Path(comp_dir) / "comparison_summary.csv"
    metrics = summarize_csv(summary_path)

    metrics["checkpoint"] = str(ckpt)
    metrics["tag"] = tag
    metrics["summary_path"] = str(summary_path)

    results.append(metrics)


out_path = Path("../my_dataset/dense_conservative_checkpoint_sweep_summary.csv")

fieldnames = [
    "checkpoint",
    "tag",
    "total_frames",
    "opencv_frames",
    "deep_frames",
    "matched_frames",
    "max_deep_corners",
    "avg_deep_corners",
    "duplicate_frames",
    "mean_error_px",
    "median_error_px",
    "good_frames",
    "borderline_frames",
    "summary_path",
]

with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print()
print("Dense conservative checkpoint sweep finished.")
print("Saved:", out_path)
print()

for r in results:
    print(
        r["tag"],
        "| deep:", r["deep_frames"],
        "| matched:", r["matched_frames"],
        "| avg:", round(r["avg_deep_corners"], 3),
        "| mean:", round(r["mean_error_px"], 3),
        "| median:", round(r["median_error_px"], 3),
        "| good:", r["good_frames"],
        "| borderline:", r["borderline_frames"],
    )
