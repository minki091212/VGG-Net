import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from dataloader import build_eval_dataset, build_loaders
from evaluate import evaluate_dense
from model import build_model, count_parameters
from trainer import Trainer


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(cfg, device):
    model = build_model(cfg["model"]).to(device)
    print(f"모델: 구성 {cfg['model']['config']}, 파라미터 {count_parameters(model):,}개")

    # 3.3절: 여러 GPU에 배치를 나눠 처리하는 데이터 병렬 학습
    if cfg["train"].get("multi_gpu") and torch.cuda.device_count() > 1:
        print(f"GPU {torch.cuda.device_count()}개로 데이터 병렬 학습")
        model = nn.DataParallel(model)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True   # 입력 크기가 고정일 때 가장 빠른 합성곱 알고리즘 선택

    train_loader, val_loader = build_loaders(cfg)
    print(f"학습 이미지 {len(train_loader.dataset):,}장, 검증 이미지 {len(val_loader.dataset):,}장")

    Trainer(model, train_loader, val_loader, cfg, device).fit()


def evaluate(cfg, device, checkpoint):
    ckpt_path = Path(checkpoint) if checkpoint else Path(cfg["output_dir"]) / "best.pth"
    model = build_model(cfg["model"])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    print(f"체크포인트: {ckpt_path} (epoch {ckpt['epoch']})")

    eval_cfg = cfg["eval"]
    dataset = build_eval_dataset(cfg)
    print(f"조밀한 평가: Q = {eval_cfg['scales']}, 좌우 뒤집기 = {eval_cfg['flip']}, 이미지 {len(dataset):,}장")

    result = evaluate_dense(model, dataset, device, scales=eval_cfg["scales"], flip=eval_cfg["flip"],
                            num_workers=cfg["data"]["num_workers"])
    print(f"top-1 오류 {result['top1_err']:.2f}%  top-5 오류 {result['top5_err']:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="VGG-Net 학습 및 평가")
    parser.add_argument("--config", required=True, help="설정 파일 경로 (예: config/vgg16.yaml)")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--checkpoint", default=None, help="평가할 체크포인트 (기본: output_dir/best.pth)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = get_device()
    print(f"장치: {device}")

    if args.mode == "train":
        train(cfg, device)
    else:
        evaluate(cfg, device, args.checkpoint)


if __name__ == "__main__":
    main()