import random

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF


class RandomShorterSideResize:
    """
    3.1절 학습 스케일 S: 이미지의 짧은 변을 [s_min, s_max]에서 무작위로 뽑은 S로 맞춘다.
    s_min == s_max이면 고정 S(단일 스케일 학습), 다르면 스케일 지터링(다중 스케일 학습)
    """

    def __init__(self, s_min, s_max):
        self.s_min, self.s_max = s_min, s_max

    def __call__(self, img):
        s = random.randint(self.s_min, self.s_max)
        return TF.resize(img, s, antialias=True)   # 정수 하나를 주면 짧은 변이 s가 됨


class ColorShift:
    """
    무작위 RGB 색상 변화 (Krizhevsky et al., 2012의 PCA 기반 방식).
    ImageNet 픽셀의 RGB 주성분 방향으로 무작위 크기만큼 색을 이동시킨다.
    입력: ToTensor 이후의 텐서 (3, H, W), 값 범위 0~1
    """

    EIGVAL = torch.tensor([0.2175, 0.0188, 0.0045])
    EIGVEC = torch.tensor([[-0.5675, 0.7192, 0.4009],
                           [-0.5808, -0.0045, -0.8140],
                           [-0.5836, -0.6948, 0.4203]])

    def __init__(self, alpha_std=0.1):
        self.alpha_std = alpha_std

    def __call__(self, img):
        alpha = torch.randn(3) * self.alpha_std
        shift = self.EIGVEC @ (alpha * self.EIGVAL)   # 채널별 이동량 (3,)
        return img + shift.view(3, 1, 1)


def build_train_transform(data_cfg):
    """학습: 짧은 변을 S로 -> 224x224 무작위 크롭 -> 좌우 뒤집기 -> RGB 색상 변화 -> 정규화"""
    t = [
        RandomShorterSideResize(data_cfg["scale_min"], data_cfg["scale_max"]),
        transforms.RandomCrop(data_cfg["crop_size"]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    if data_cfg.get("color_shift", 0) > 0:
        t.append(ColorShift(data_cfg["color_shift"]))
    t.append(transforms.Normalize(data_cfg["mean"], data_cfg["std"]))
    return transforms.Compose(t)


def build_val_transform(data_cfg):
    """
    학습 중 검증용: 짧은 변을 Q로 -> 중앙 224x224 크롭.
    크기를 맞춰야 배치로 묶을 수 있으므로 크롭을 사용한다. (최종 평가는 evaluate.py의 조밀한 평가)
    """
    return transforms.Compose([
        transforms.Resize(data_cfg["val_scale"], antialias=True),
        transforms.CenterCrop(data_cfg["crop_size"]),
        transforms.ToTensor(),
        transforms.Normalize(data_cfg["mean"], data_cfg["std"]),
    ])


def build_full_image_transform(data_cfg):
    """조밀한 평가용: 크기를 바꾸지 않고 정규화만 한다. 크기 조정은 평가 시 Q마다 수행"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_cfg["mean"], data_cfg["std"]),
    ])


def build_loaders(cfg):
    """학습용, 검증용 DataLoader 생성. 데이터 폴더는 클래스별 하위 폴더 구조(ImageFolder)여야 한다"""
    data_cfg, train_cfg = cfg["data"], cfg["train"]

    train_set = datasets.ImageFolder(data_cfg["train_dir"], transform=build_train_transform(data_cfg))
    val_set = datasets.ImageFolder(data_cfg["val_dir"], transform=build_val_transform(data_cfg))

    train_loader = DataLoader(
        train_set,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=data_cfg["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    return train_loader, val_loader


def build_eval_dataset(cfg):
    """조밀한 평가용 데이터셋 (이미지마다 크기가 다르므로 배치 크기 1로 사용)"""
    data_cfg = cfg["data"]
    return datasets.ImageFolder(data_cfg["val_dir"], transform=build_full_image_transform(data_cfg))