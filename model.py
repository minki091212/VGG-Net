import torch
import torch.nn as nn

# 표 1의 구성. 정수 = 3x3 합성곱 출력 채널, (채널, 1) = 1x1 합성곱, "M" = max-pooling
CFGS = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "C": [64, 64, "M", 128, 128, "M", 256, 256, (256, 1), "M",
          512, 512, (512, 1), "M", 512, 512, (512, 1), "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
          512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M",
          512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def make_features(cfg, batch_norm=False):
    layers, in_ch = [], 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue
        out_ch, k = v if isinstance(v, tuple) else (v, 3)
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=1, padding=k // 2))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        in_ch = out_ch
    return nn.Sequential(*layers)


class VGG(nn.Module):
    """
    config: "A" | "B" | "C" | "D"(VGG16) | "E"(VGG19)
    head:   "fc"  = 논문 그대로 FC-4096-4096-클래스 수 (학습 입력은 224x224)
            "gap" = global average pooling + FC 1개 (가벼운 버전, 입력 크기 자유)
    """

    def __init__(self, config="D", num_classes=1000, batch_norm=True, head="fc", dropout=0.5):
        super().__init__()
        self.head = head
        self.features = make_features(CFGS[config], batch_norm)

        if head == "fc":
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(4096, num_classes),
            )
        elif head == "gap":
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(512, num_classes),
            )
        else:
            raise ValueError(f"unknown head: {head}")

        self._init_weights()

    def _init_weights(self):
        # 논문의 단계적 사전 초기화 대신 He 초기화 사용
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # 반환값은 soft-max 이전의 점수(logits). 손실은 nn.CrossEntropyLoss 사용
        return self.classifier(self.features(x))

    @torch.no_grad()
    def to_fully_conv(self):
        """
        3.2절의 조밀한 평가용: FC 층을 같은 가중치의 합성곱 층으로 변환
        FC1 -> 7x7 합성곱, FC2·FC3 -> 1x1 합성곱
        """
        if self.head != "fc":
            raise RuntimeError("head='fc'일 때만 변환할 수 있습니다.")
        fc1, fc2, fc3 = [m for m in self.classifier if isinstance(m, nn.Linear)]

        conv1 = nn.Conv2d(512, fc1.out_features, kernel_size=7)
        conv2 = nn.Conv2d(fc2.in_features, fc2.out_features, kernel_size=1)
        conv3 = nn.Conv2d(fc3.in_features, fc3.out_features, kernel_size=1)

        # nn.Flatten은 (채널, 높이, 너비) 순서로 펼치므로 같은 순서로 다시 접으면 된다
        conv1.weight.copy_(fc1.weight.view(fc1.out_features, 512, 7, 7))
        conv2.weight.copy_(fc2.weight.view(fc2.out_features, fc2.in_features, 1, 1))
        conv3.weight.copy_(fc3.weight.view(fc3.out_features, fc3.in_features, 1, 1))
        for conv, fc in ((conv1, fc1), (conv2, fc2), (conv3, fc3)):
            conv.bias.copy_(fc.bias)

        return nn.Sequential(
            self.features,
            conv1, nn.ReLU(inplace=True),
            conv2, nn.ReLU(inplace=True),
            conv3,
        )


@torch.no_grad()
def dense_predict(fully_conv_net, images):
    """
    이미지 전체를 넣어 클래스 점수 맵을 만들고, 공간 평균으로 클래스 점수 벡터를 얻는다.
    원본과 좌우 뒤집은 이미지의 soft-max 확률을 평균한다.
    images: (N, 3, H, W), H와 W는 224 이상
    """
    fully_conv_net.eval()
    probs = 0
    for x in (images, torch.flip(images, dims=[3])):
        score_map = fully_conv_net(x)            # (N, 클래스 수, H', W')
        scores = score_map.mean(dim=(2, 3))      # 공간 평균 -> (N, 클래스 수)
        probs = probs + scores.softmax(dim=1)
    return probs / 2


def vgg16(**kwargs):
    return VGG("D", **kwargs)


def vgg19(**kwargs):
    return VGG("E", **kwargs)