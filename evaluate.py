import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import DenseClassifier


@torch.no_grad()
def topk_correct(logits, targets, ks=(1, 5)):
    """각 k에 대해, 정답이 상위 k개 예측 안에 든 샘플 수를 반환"""
    maxk = min(max(ks), logits.size(1))
    pred = logits.topk(maxk, dim=1).indices            # (N, maxk)
    hit = pred.eq(targets.view(-1, 1))                 # (N, maxk), 정답 위치만 True
    return [hit[:, :min(k, maxk)].any(dim=1).sum().item() for k in ks]


@torch.no_grad()
def evaluate_crop(model, loader, device, criterion=None, use_amp=False):
    """
    학습 중 검증: 중앙 크롭 이미지를 배치로 평가.
    반환: 평균 손실, top-1 오류(%), top-5 오류(%)
    """
    model.eval()
    total, loss_sum, c1, c5 = 0, 0.0, 0, 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            if criterion is not None:
                loss_sum += criterion(logits, targets).item() * images.size(0)
        top1, top5 = topk_correct(logits.float(), targets)
        c1, c5, total = c1 + top1, c5 + top5, total + images.size(0)

    return {
        "loss": loss_sum / total if criterion is not None else None,
        "top1_err": 100.0 * (1 - c1 / total),
        "top5_err": 100.0 * (1 - c5 / total),
    }


def resize_shorter_side(x, q):
    """텐서 (1, 3, H, W)의 짧은 변을 q로 맞춘다 (가로세로 비율 유지)"""
    h, w = x.shape[-2:]
    scale = q / min(h, w)
    size = (max(round(h * scale), q), max(round(w * scale), q))
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=True)


@torch.no_grad()
def evaluate_dense(model, dataset, device, scales=(256, 384, 512), flip=True,
                   num_workers=4, log_interval=1000):
    """
    3.2절, 4.2절의 평가 방식.
    1) 이미지의 짧은 변을 각 Q로 맞춘다 (다중 스케일)
    2) 크롭하지 않은 전체 이미지를 완전 합성곱 네트워크에 넣고, 점수 맵을 공간 평균한다
    3) 좌우 뒤집은 이미지도 같은 방식으로 처리한다
    4) 모든 결과의 soft-max 확률을 평균해 최종 예측으로 쓴다
    """
    dense = DenseClassifier(model).to(device).eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

    total, c1, c5 = 0, 0, 0
    for i, (image, target) in enumerate(loader, start=1):
        image, target = image.to(device), target.to(device)

        probs, n = 0, 0
        for q in scales:
            x = resize_shorter_side(image, q)
            views = (x, torch.flip(x, dims=[3])) if flip else (x,)
            for v in views:
                probs = probs + dense(v).softmax(dim=1)
                n += 1
        probs = probs / n

        top1, top5 = topk_correct(probs, target)
        c1, c5, total = c1 + top1, c5 + top5, total + 1

        if i % log_interval == 0:
            print(f"[eval] {i}/{len(loader)}  top-1 err {100 * (1 - c1 / total):.2f}%  "
                  f"top-5 err {100 * (1 - c5 / total):.2f}%")

    return {
        "top1_err": 100.0 * (1 - c1 / total),
        "top5_err": 100.0 * (1 - c5 / total),
    }