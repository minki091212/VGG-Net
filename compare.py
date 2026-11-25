"""
이미 학습이 끝난 실험들을 비교하는 표와 그래프를 한 번에 만든다.
기존 파일(log.csv, best.pth)은 읽기만 하고 바꾸지 않는다.

사용법
  # 표와 그래프만 (log.csv 사용)
  uv run python compare.py --configs config/Vgg11.yaml config/Vgg16.yaml

  # 다중 스케일 조밀한 평가까지 포함 (best.pth로 평가를 실행하고 결과를 results/에 저장)
  uv run python compare.py --configs config/Vgg11.yaml config/Vgg16.yaml --eval

  # 학습 시간을 알고 있으면 직접 입력 (시간 단위, configs 순서대로)
  uv run python compare.py --configs config/Vgg11.yaml config/Vgg16.yaml --hours 1.5 3.2

결과 (results/ 폴더)
  summary.md, summary.csv   비교표
  comparison.png            그래프 6개를 한 장에 모은 그림
  val_top1_err.png 등       그래프 개별 파일
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # 화면 없이 파일로만 저장
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullFormatter
import yaml

from model import CFGS

# ---------------------------------------------------------------- 색과 스타일
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # 모델 순서대로 고정
SURFACE = "#fcfcfb"
TEXT = "#2b2b29"
TEXT_MUTED = "#6b6a64"
GRID = "#e4e3de"

KOREAN_FONTS = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR",
                "Noto Sans CJK JP"]

LABELS = {
    "ko": {
        "val_err": "검증 top-1 오류 (중앙 크롭)",
        "loss": "학습 손실과 검증 손실",
        "lr": "학습률",
        "crop_dense": "평가 방식별 top-1 오류",
        "params": "파라미터 수",
        "macs": "이미지 1장당 연산량",
        "epoch": "에폭",
        "err": "오류 (%)",
        "loss_y": "교차 엔트로피 손실",
        "params_y": "파라미터 (백만 개)",
        "macs_y": "곱셈-덧셈 (십억 회)",
        "train": "학습",
        "val": "검증",
        "best": "최저",
        "crop": "중앙 크롭 (학습 중 최고)",
        "dense": "다중 스케일 조밀한 평가",
        "no_dense": "--eval 옵션으로 실행하면\n다중 스케일 결과가 추가됩니다",
        "title": "깊이에 따른 성능 비교",
    },
    "en": {
        "val_err": "Validation top-1 error (center crop)",
        "loss": "Training vs validation loss",
        "lr": "Learning rate",
        "crop_dense": "Top-1 error by evaluation method",
        "params": "Parameters",
        "macs": "Compute per image",
        "epoch": "Epoch",
        "err": "Error (%)",
        "loss_y": "Cross-entropy loss",
        "params_y": "Parameters (millions)",
        "macs_y": "Multiply-adds (billions)",
        "train": "train",
        "val": "val",
        "best": "min",
        "crop": "Center crop (best during training)",
        "dense": "Multi-scale dense",
        "no_dense": "Run with --eval to add\nmulti-scale results",
        "title": "Effect of depth",
    },
}


def setup_style():
    """한글 글꼴이 있으면 한글, 없으면 영어로 그래프 글자를 표시"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    lang = "en"
    for name in KOREAN_FONTS:
        if name in available:
            plt.rcParams["font.family"] = name
            lang = "ko"
            break
    if lang == "en":
        print("한글 글꼴을 찾지 못해 그래프 글자를 영어로 표시합니다.")

    plt.rcParams.update({
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": TEXT,
    })
    return LABELS[lang]


# ---------------------------------------------------------------- 모델 정보 (계산으로 구함, GPU 불필요)
def model_stats(model_cfg, size=224):
    """구성표로 깊이, 파라미터 수, 이미지 1장당 곱셈-덧셈 횟수를 계산"""
    bn = model_cfg.get("batch_norm", False)
    nc = model_cfg["num_classes"]
    h = w = size
    in_ch, params, macs, n_conv = 3, 0, 0, 0
    for v in CFGS[model_cfg["config"]]:
        if v == "M":
            h, w = h // 2, w // 2
            continue
        out_ch, k = v if isinstance(v, tuple) else (v, 3)
        params += k * k * in_ch * out_ch + (2 * out_ch if bn else out_ch)  # 가중치 + (BN 2개 또는 편향)
        macs += h * w * k * k * in_ch * out_ch
        in_ch, n_conv = out_ch, n_conv + 1

    if model_cfg.get("head", "fc") == "fc":
        fcs = [(512 * 7 * 7, 4096), (4096, 4096), (4096, nc)]
    else:
        fcs = [(512, nc)]
    for i, o in fcs:
        params += i * o + o
        macs += i * o
    return {"depth": n_conv + len(fcs), "params": params, "macs": macs}


# ---------------------------------------------------------------- 결과 읽기
def read_log(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({k: (int(v) if k == "epoch" else float(v)) for k, v in r.items()})
    return rows


def run_dense_eval(cfg, cache_path):
    """best.pth로 다중 스케일 조밀한 평가를 실행하고 결과를 results/ 폴더에 저장"""
    import torch
    from dataloader import build_eval_dataset
    from evaluate import evaluate_dense
    from main import get_device
    from model import build_model

    device = get_device()
    ckpt_path = Path(cfg["output_dir"]) / "best.pth"
    model = build_model(cfg["model"])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    print(f"조밀한 평가 실행: {ckpt_path} (epoch {ckpt['epoch']})")

    result = evaluate_dense(model, build_eval_dataset(cfg), device,
                            scales=cfg["eval"]["scales"], flip=cfg["eval"]["flip"],
                            num_workers=cfg["data"]["num_workers"])
    result.update({"checkpoint": str(ckpt_path), "epoch": ckpt["epoch"], "scales": cfg["eval"]["scales"]})
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def load_run(config_path, out_dir, do_eval, color):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_dir = Path(cfg["output_dir"])
    log_path = run_dir / "log.csv"
    if not log_path.exists():
        raise FileNotFoundError(f"{log_path}가 없습니다. 학습 결과 폴더(output_dir)를 확인하세요.")

    stats = model_stats(cfg["model"])
    name = Path(config_path).stem
    run = {
        "name": name,
        "label": f"VGG{stats['depth']} ({cfg['model']['config']})",
        "color": color,
        "rows": read_log(log_path),
        **stats,
    }

    # 다중 스케일 평가 결과: results/에 저장된 것이 있으면 사용, --eval이면 새로 실행
    cache_path = out_dir / f"eval_{name}.json"
    run["dense"] = None
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            run["dense"] = json.load(f)
    elif do_eval:
        run["dense"] = run_dense_eval(cfg, cache_path)
    return run


# ---------------------------------------------------------------- 요약 계산
def summarize(run, hours=None):
    rows = run["rows"]
    best = min(rows, key=lambda r: r["val_top1_err"])
    lr_drops = [rows[i]["epoch"] for i in range(1, len(rows)) if rows[i]["lr"] < rows[i - 1]["lr"]]
    s = {
        "모델": run["label"],
        "깊이(층)": run["depth"],
        "파라미터(M)": round(run["params"] / 1e6, 2),
        "연산량(GMACs)": round(run["macs"] / 1e9, 2),
        "학습 에폭": len(rows),
        "최고 에폭": best["epoch"],
        "검증 top-1 오류(%)": best["val_top1_err"],
        "검증 top-5 오류(%)": best["val_top5_err"],
        "학습 손실": best["train_loss"],
        "검증 손실": best["val_loss"],
        "손실 차이(검증-학습)": round(best["val_loss"] - best["train_loss"], 4),
        "학습률 감소 에폭": ", ".join(map(str, lr_drops)) if lr_drops else "-",
        "다중 스케일 top-1(%)": "-",
        "다중 스케일 top-5(%)": "-",
        "다중 스케일 개선(%p)": "-",
    }
    if run["dense"]:
        s["다중 스케일 top-1(%)"] = round(run["dense"]["top1_err"], 2)
        s["다중 스케일 top-5(%)"] = round(run["dense"]["top5_err"], 2)
        s["다중 스케일 개선(%p)"] = round(best["val_top1_err"] - run["dense"]["top1_err"], 2)
    if hours is not None:
        s["학습 시간(h)"] = hours
    return s


def write_tables(summaries, out_dir):
    keys = list(summaries[0].keys())
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)

    # 리뷰에 바로 붙일 수 있게 행 = 항목, 열 = 모델 형태로 작성
    lines = ["| 항목 | " + " | ".join(str(s["모델"]) for s in summaries) + " |",
             "|---|" + "---|" * len(summaries)]
    for k in keys[1:]:
        lines.append(f"| {k} | " + " | ".join(str(s[k]) for s in summaries) + " |")
    notes = [
        "",
        "- 검증 top-1/top-5 오류, 손실은 학습 중 검증(중앙 224×224 크롭)에서 top-1 오류가 가장 낮았던 에폭의 값",
        "- 학습 손실은 데이터 증강과 드롭아웃이 적용된 상태에서 계산되므로, 검증 손실과의 차이는 과적합의 대략적인 지표",
        "- 다중 스케일 개선 = 중앙 크롭 최고 top-1 오류 − 다중 스케일 조밀한 평가 top-1 오류",
        "- 연산량은 224×224 입력 기준",
    ]
    text = "\n".join(lines + notes) + "\n"
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + "\n".join(lines))


# ---------------------------------------------------------------- 그래프
def plot_val_err(ax, runs, L):
    for r in runs:
        ep = [x["epoch"] for x in r["rows"]]
        err = [x["val_top1_err"] for x in r["rows"]]
        best = min(r["rows"], key=lambda x: x["val_top1_err"])
        # 최저 오류 값은 범례에 함께 표시 (그래프 위 글자가 선과 겹치지 않도록)
        ax.plot(ep, err, color=r["color"], linewidth=2,
                label=f"{r['label']} · {L['best']} {best['val_top1_err']:.2f}% (epoch {best['epoch']})")
        ax.plot(best["epoch"], best["val_top1_err"], "o", markersize=8, color=r["color"],
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    ax.set(title=L["val_err"], xlabel=L["epoch"], ylabel=L["err"])
    ax.legend()


def plot_loss(ax, runs, L):
    for r in runs:
        ep = [x["epoch"] for x in r["rows"]]
        ax.plot(ep, [x["train_loss"] for x in r["rows"]], color=r["color"], linewidth=2)
        ax.plot(ep, [x["val_loss"] for x in r["rows"]], color=r["color"], linewidth=2, linestyle="--")
    model_handles = [Line2D([], [], color=r["color"], linewidth=2, label=r["label"]) for r in runs]
    style_handles = [Line2D([], [], color=TEXT_MUTED, linewidth=2, label=L["train"]),
                     Line2D([], [], color=TEXT_MUTED, linewidth=2, linestyle="--", label=L["val"])]
    ax.legend(handles=model_handles + style_handles, ncol=2)
    ax.set(title=L["loss"], xlabel=L["epoch"], ylabel=L["loss_y"])


def plot_lr(ax, runs, L):
    for r in runs:
        ep = [x["epoch"] for x in r["rows"]]
        ax.step(ep, [x["lr"] for x in r["rows"]], where="post", color=r["color"], linewidth=2, label=r["label"])
    ax.set_yscale("log")
    # 기본 눈금(10^-2 형태)은 수식 글꼴을 써서 Windows에서 '−' 글자가 깨지므로 일반 숫자로 표시
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set(title=L["lr"], xlabel=L["epoch"])
    ax.legend()


def plot_crop_dense(ax, runs, L):
    width = 0.36
    has_dense = any(r["dense"] for r in runs)
    for i, r in enumerate(runs):
        best = min(x["val_top1_err"] for x in r["rows"])
        x_crop = i - width / 2 if has_dense else i
        b = ax.bar(x_crop, best, width, facecolor=SURFACE, edgecolor=r["color"], hatch="//", linewidth=1.5)
        ax.bar_label(b, fmt="%.2f", padding=2, fontsize=9, color=TEXT)
        if r["dense"]:
            b = ax.bar(i + width / 2, r["dense"]["top1_err"], width, color=r["color"],
                       edgecolor=SURFACE, linewidth=2)
            ax.bar_label(b, fmt="%.2f", padding=2, fontsize=9, color=TEXT)
    ax.set_xticks(range(len(runs)), [r["label"] for r in runs])
    ax.set(title=L["crop_dense"], ylabel=L["err"])
    ax.margins(y=0.18)
    ax.grid(axis="x", visible=False)
    handles = [Patch(facecolor=SURFACE, edgecolor=TEXT_MUTED, hatch="//", label=L["crop"])]
    if has_dense:
        handles.append(Patch(facecolor=TEXT_MUTED, label=L["dense"]))
    else:
        ax.text(0.5, 0.5, L["no_dense"], transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color=TEXT_MUTED)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)


def _plot_bars(ax, runs, values, fmt, title, ylabel):
    b = ax.bar(range(len(runs)), values, 0.5, color=[r["color"] for r in runs], edgecolor=SURFACE, linewidth=2)
    ax.bar_label(b, fmt=fmt, padding=2, fontsize=9, color=TEXT)
    ax.set_xticks(range(len(runs)), [r["label"] for r in runs])
    ax.set(title=title, ylabel=ylabel)
    ax.margins(y=0.15)
    ax.grid(axis="x", visible=False)


def plot_params(ax, runs, L):
    _plot_bars(ax, runs, [r["params"] / 1e6 for r in runs], "%.1f", L["params"], L["params_y"])


def plot_macs(ax, runs, L):
    _plot_bars(ax, runs, [r["macs"] / 1e9 for r in runs], "%.2f", L["macs"], L["macs_y"])


PANELS = [
    ("val_top1_err", plot_val_err),
    ("loss", plot_loss),
    ("learning_rate", plot_lr),
    ("crop_vs_dense", plot_crop_dense),
    ("params", plot_params),
    ("compute", plot_macs),
]


def save_figures(runs, L, out_dir):
    # 한 장에 모은 그림
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for ax, (_, fn) in zip(axes.flat, PANELS):
        fn(ax, runs, L)
    fig.suptitle(L["title"] + ": " + " vs ".join(r["label"] for r in runs),
                 fontsize=15, fontweight="bold", color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=3)
    fig.savefig(out_dir / "comparison.png", dpi=200)
    plt.close(fig)

    # 개별 그림
    for name, fn in PANELS:
        fig, ax = plt.subplots(figsize=(7, 4.6))
        fn(ax, runs, L)
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}.png", dpi=200)
        plt.close(fig)


# ---------------------------------------------------------------- 실행
def main():
    p = argparse.ArgumentParser(description="학습 결과 비교 표와 그래프 생성")
    p.add_argument("--configs", nargs="+", required=True, help="비교할 설정 파일들")
    p.add_argument("--out", default="results", help="결과 저장 폴더")
    p.add_argument("--eval", action="store_true", help="best.pth로 다중 스케일 조밀한 평가 실행 (GPU 권장)")
    p.add_argument("--hours", nargs="+", type=float, default=None, help="모델별 학습 시간(시간), configs 순서대로")
    args = p.parse_args()

    if args.hours and len(args.hours) != len(args.configs):
        p.error("--hours 개수는 --configs 개수와 같아야 합니다.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    L = setup_style()

    runs = [load_run(c, out_dir, args.eval, SERIES[i % len(SERIES)]) for i, c in enumerate(args.configs)]
    hours = args.hours or [None] * len(runs)
    write_tables([summarize(r, h) for r, h in zip(runs, hours)], out_dir)
    save_figures(runs, L, out_dir)
    print(f"\n저장 완료: {out_dir.resolve()}")


if __name__ == "__main__":
    main()