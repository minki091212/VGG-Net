# VGG-Net

Simonyan & Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition" (ICLR 2015)의 PyTorch 구현

## 설치

```bash
uv add torch torchvision pyyaml
```

## 파일 구성

| 파일 | 역할 |
|---|---|
| `model.py` | VGG 구성 A–E(표 1), FC 층의 합성곱 변환(3.2절) |
| `dataloader.py` | 학습용 전처리(스케일 지터링, 무작위 크롭, 뒤집기, RGB 색상 변화), DataLoader |
| `trainer.py` | 학습 루프, 학습률 감소, 체크포인트 저장 |
| `evaluate.py` | top-1/top-5 오류 계산, 다중 스케일 조밀한 평가 |
| `main.py` | 실행 진입점 |
| `config/Vgg16.yaml` | 구성 D (VGG16) 설정 |
| `config/Vgg19.yaml` | 구성 E (VGG19) 설정 |

## 데이터 준비

Imagenette(ImageNet 중 10개 클래스)를 사용합니다. 프로젝트 폴더에서 한 번만 실행합니다.

```bash
uv run python -c "from torchvision.datasets import Imagenette; Imagenette('data', download=True)"
```

다운로드가 끝나면 다음 구조가 생깁니다.

```
data/imagenette2/
├── train/   (클래스별 폴더 10개)
└── val/     (클래스별 폴더 10개)
```

## 실행

```bash
# 학습
uv run python main.py --config config/Vgg16.yaml
uv run python main.py --config config/Vgg19.yaml

# 이어서 학습: 설정 파일의 train.resume에 runs/vgg16/last.pth 지정 후 다시 실행

# 평가 (다중 스케일 조밀한 평가, 기본 체크포인트: output_dir/best.pth)
uv run python main.py --config config/Vgg16.yaml --mode eval
uv run python main.py --config config/Vgg16.yaml --mode eval --checkpoint runs/vgg16/last.pth
```

학습 결과는 `output_dir`(예: `runs/vgg16/`)에 저장됩니다.

- `last.pth`: 마지막 에폭의 체크포인트
- `best.pth`: 검증 정확도가 가장 높았던 체크포인트
- `log.csv`: 에폭별 손실, 오류율, 학습률

## 논문과 다른 점

| 항목 | 논문 | 이 구현 |
|---|---|---|
| 데이터 | ImageNet (1000클래스, 학습 128만 장) | Imagenette (10클래스, 학습 9,469장) |
| BatchNorm | 없음 | 사용 (데이터가 적고 처음부터 학습하므로) |
| 가중치 초기화 | 구성 A를 먼저 학습한 뒤 그 가중치로 깊은 구성 초기화 | He 초기화 (단계적 사전 학습 없음) |
| 정규화 | 평균 RGB만 뺌 | 평균·표준편차로 정규화 (설정에서 std를 1로 두면 논문과 같음) |
| 배치 크기 | 256 | 64 |
| 학습 중 검증 | 명시되지 않음 | Q로 크기 조정 후 중앙 224×224 크롭 |
| 다중 크롭 평가 (4.3절) | 스케일당 50개 크롭 | 구현하지 않음 (조밀한 평가만) |
| 다중 스케일 모델 학습 | S=384 모델을 미세 조정 | 처음부터 스케일 지터링으로 학습 |
| 연산 정밀도 | 32비트 | 혼합 정밀도(AMP) 선택 가능 |