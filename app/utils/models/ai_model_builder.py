# make_deeplab_onnx.py — DeepLab v3 ONNX ビルドスクリプト（手動実行専用・import 時は何もしない）
# 実行には torch/torchvision が必要（requirements 未宣言の重い依存のため遅延 import）
#
# 使い方: venv/bin/python -m app.utils.models.ai_model_builder
# 生成物: deeplabv3_mnv3.onnx（実行ディレクトリ直下）


def build_deeplab_onnx() -> str:
    """学習済み DeepLab v3 (MobileNetV3 Large) を ONNX にエクスポートする."""
    import torch
    import torchvision

    # 1) 学習済み DeepLab v3 + MobileNetV3 Large（Cityscapes 21クラス）
    model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights="DEFAULT")  # PyTorch 2.2 時点で公開[3]
    model.eval()

    # 2) ダミー入力（513×513 は元論文の標準解像度）
    dummy = torch.randn(1, 3, 513, 513)

    # 3) ONNX にエクスポート
    torch.onnx.export(
        model,
        dummy,
        "deeplabv3_mnv3.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=12,  # OpenCV 4.7+ が正式対応[3]
    )
    print("✓ deeplabv3_mnv3.onnx を生成しました")
    return "deeplabv3_mnv3.onnx"


if __name__ == "__main__":
    build_deeplab_onnx()
