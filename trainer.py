import csv
import time
from pathlib import Path

import torch
import torch.nn as nn

from evaluate import evaluate_crop, topk_correct


def unwrap(model):
    """DataParallel로 감싼 경우 원래 모델을 꺼낸다 (저장·불러오기 시 키 이름을 맞추기 위함)"""
    return model.module if isinstance(model, nn.DataParallel) else model


class Trainer:
    """
    3.1절 학습 절차
    - 다항 로지스틱 회귀(교차 엔트로피) + 모멘텀 SGD
    - 배치 256, 모멘텀 0.9, 가중치 감쇠 5e-4
    - 학습률 1e-2에서 시작, 검증 정확도가 멈추면 1/10로 감소
    - 학습률을 정해진 횟수만큼 줄인 뒤 다시 멈추면 학습 종료
    """

    def __init__(self, model, train_loader, val_loader, cfg, device):
        t = cfg["train"]
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            model.parameters(), lr=t["lr"], momentum=t["momentum"], weight_decay=t["weight_decay"]
        )
        # mode="max": 검증 정확도가 patience 에폭 동안 오르지 않으면 학습률에 factor를 곱함
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=t["lr_factor"], patience=t["lr_patience"]
        )

        self.epochs = t["epochs"]
        self.max_lr_drops = t["max_lr_drops"]
        self.log_interval = t.get("log_interval", 100)

        # 혼합 정밀도(AMP): GPU에서 일부 연산을 16비트로 수행해 속도와 메모리를 절약
        self.use_amp = t.get("amp", False) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.out_dir = Path(cfg["output_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_dir / "log.csv"

        self.start_epoch = 1
        self.best_acc = 0.0
        self.lr_drops = 0
        if t.get("resume"):
            self.load_checkpoint(t["resume"])

    # ---------------------------------------------------------------- 학습
    def train_one_epoch(self, epoch):
        self.model.train()
        total, loss_sum, correct = 0, 0.0, 0
        start = time.time()

        for step, (images, targets) in enumerate(self.train_loader, start=1):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            n = images.size(0)
            total += n
            loss_sum += loss.item() * n
            correct += topk_correct(logits.detach().float(), targets, ks=(1,))[0]

            if step % self.log_interval == 0:
                print(f"[epoch {epoch}] step {step}/{len(self.train_loader)}  "
                      f"loss {loss_sum / total:.4f}  top-1 acc {100 * correct / total:.2f}%  "
                      f"{time.time() - start:.0f}s")

        return {"loss": loss_sum / total, "top1_acc": 100.0 * correct / total}

    def fit(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            train_stats = self.train_one_epoch(epoch)
            val_stats = evaluate_crop(self.model, self.val_loader, self.device,
                                      criterion=self.criterion, use_amp=self.use_amp)
            val_acc = 100.0 - val_stats["top1_err"]

            # 학습률 감소 여부 확인
            lr_before = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_acc)
            lr_after = self.optimizer.param_groups[0]["lr"]
            if lr_after < lr_before:
                self.lr_drops += 1
                print(f"학습률 감소 ({self.lr_drops}회): {lr_before:g} -> {lr_after:g}")

            is_best = val_acc > self.best_acc
            self.best_acc = max(self.best_acc, val_acc)

            print(f"[epoch {epoch}] train loss {train_stats['loss']:.4f} | "
                  f"val loss {val_stats['loss']:.4f}  top-1 err {val_stats['top1_err']:.2f}%  "
                  f"top-5 err {val_stats['top5_err']:.2f}% | lr {lr_after:g}")

            self.write_log(epoch, train_stats, val_stats, lr_after)
            self.save_checkpoint(epoch, self.out_dir / "last.pth")
            if is_best:
                self.save_checkpoint(epoch, self.out_dir / "best.pth")

            # 논문: 학습률을 3번 줄인 뒤 학습 종료
            if self.lr_drops > self.max_lr_drops:
                print(f"학습률을 {self.max_lr_drops}번 줄인 뒤 다시 정체되어 학습을 종료합니다.")
                break

        print(f"학습 완료. 최고 검증 top-1 정확도 {self.best_acc:.2f}%")

    # ---------------------------------------------------------------- 기록·저장
    def write_log(self, epoch, train_stats, val_stats, lr):
        new_file = not self.log_path.exists()
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["epoch", "lr", "train_loss", "train_top1_acc",
                                 "val_loss", "val_top1_err", "val_top5_err"])
            writer.writerow([epoch, lr, f"{train_stats['loss']:.4f}", f"{train_stats['top1_acc']:.2f}",
                             f"{val_stats['loss']:.4f}", f"{val_stats['top1_err']:.2f}",
                             f"{val_stats['top5_err']:.2f}"])

    def save_checkpoint(self, epoch, path):
        torch.save({
            "epoch": epoch,
            "model": unwrap(self.model).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_acc": self.best_acc,
            "lr_drops": self.lr_drops,
            "config": self.cfg,
        }, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        unwrap(self.model).load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.best_acc = ckpt["best_acc"]
        self.lr_drops = ckpt["lr_drops"]
        self.start_epoch = ckpt["epoch"] + 1
        print(f"체크포인트 불러옴: {path} (epoch {ckpt['epoch']}부터 이어서 학습)")