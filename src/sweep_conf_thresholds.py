from pathlib import Path
import subprocess
import re
import csv
import math


MODEL_UTILS = Path("models/model_utils.py")


def patch_thresholds(loc_thresh, id_thresh):
    txt = MODEL_UTILS.read_text()

    txt = re.sub(
        r"loc_conf_thresh: float = [0-9.]+",
        f"loc_conf_thresh: float = {loc_thresh}",
        txt
    )

    txt = re.sub(
        r"id_conf_thresh: float = [0-9.]+",
        f"id_conf_thresh: float = {id_thresh}",
        txt
    )

    MODEL_UTILS.write_text(txt)


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
    good_frames = 0
    borderline_frames = 0

    for r in rows:
        if r["duplicate_deep_ids"] not in ["[]", "", None]:
            duplicate_frames += 1

        if r["mean_error_px"] != "":
            err = float(r["mean_error_px"])
            errors.append(err)

            m = int(r["matched_ids"])

            if m >= 3 and err < 40 and r["duplicate_deep_ids"] == "[]":
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


thresholds = [
    (0.03, 0.03),
    (0.05, 0.05),
    (0.07, 0.05),
    (0.07, 0.07),
    (0.10, 0.05),
    (0.10, 0.10),
    (0.15, 0.05),
    (0.15, 0.10),
]

results = []

for loc_t, id_t in thresholds:
    tag = f"loc{str(loc_t).replace('.', '')}_id{str(id_t).replace('.', '')}"

    print("\n==============================")
    print("Testing thresholds:", loc_t, id_t)
    print("==============================")

    patch_thresholds(loc_t, id_t)

    infer_img_dir = f"../my_dataset/inference_output_conf_sweep/{tag}/images"
    infer_kpt_dir = f"../my_dataset/inference_output_conf_sweep/{tag}/keypoints"
    comp_dir = f"../my_dataset/comparison_output_conf_sweep/{tag}"

    patch_file(
        "run_custom_inference.py",
        [
            (r'image_dir = ".*?"', 'image_dir = "../my_dataset/raw_frames"'),
            (r'output_img_dir = ".*?"', f'output_img_dir = "{infer_img_dir}"'),
            (r'output_kpt_dir = ".*?"', f'output_kpt_dir = "{infer_kpt_dir}"'),
            (
                r'deepc_path = ".*?"',
                'deepc_path = "./tb_logs/ckpts_deepcharuco_masked_pseudo_min4/best_masked_min4_epoch7.ckpt"'
            ),
            (
                r'refinenet_path = .*',
                'refinenet_path = "./tb_logs/ckpts_refinenet/epoch=193-step=9312.ckpt"'
            ),
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
    metrics["loc_thresh"] = loc_t
    metrics["id_thresh"] = id_t
    metrics["summary_path"] = str(summary_path)

    results.append(metrics)


out_path = Path("../my_dataset/conf_threshold_sweep_summary.csv")

fieldnames = [
    "loc_thresh",
    "id_thresh",
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

print("\nThreshold sweep finished.")
print("Saved:", out_path)

for r in results:
    print(
        "loc=", r["loc_thresh"],
        "id=", r["id_thresh"],
        "deep=", r["deep_frames"],
        "matched=", r["matched_frames"],
        "avg_deep=", round(r["avg_deep_corners"], 3),
        "dups=", r["duplicate_frames"],
        "mean=", r["mean_error_px"],
        "good=", r["good_frames"],
        "borderline=", r["borderline_frames"],
    )