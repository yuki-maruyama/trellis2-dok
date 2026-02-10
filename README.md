# TRELLIS.2 DOK - VRChat髪アセット生成パイプライン

Microsoft TRELLIS.2 を使った超高速3D生成パイプライン（さくらクラウド 高火力DOK対応）

## 特徴

- ⚡ **超高速**: 512³解像度で3秒、1024³で17秒（H100）
- 🎨 **PBRマテリアル**: Base Color, Metallic, Roughness, Opacity完全対応
- ✅ **複雑なトポロジー**: VRChat髪アセットに最適
- 🚀 **単一画像から生成**: Image-to-3D

## 性能

| 解像度 | H100 | V100（予想） |
|--------|------|--------------|
| 512³ | 3秒 | 10秒 |
| 1024³ | 17秒 | 50秒 |

**vs InstantMesh**: 7分41秒 → 10秒（**95%短縮**）

## 使い方

```bash
curl -X POST "https://secure.sakura.ad.jp/cloud/zone/is1a/api/managed-container/1.0/tasks/" \
  -u "$API_KEY:$API_SECRET" \
  -H "Content-Type: application/json" \
  -d '{
    "containers": [{
      "plan": "v100-32gb",
      "image": "ghcr.io/yuki-maruyama/trellis2-dok:latest",
      "environment": {
        "IMAGE_URL": "https://example.com/image.jpg",
        "RESOLUTION": "512"
      }
    }]
  }'
```

## 環境変数

- `IMAGE_URL`: 入力画像URL（必須）
- `RESOLUTION`: 出力解像度（512/1024/1536、デフォルト: 512）

## 出力

- `output.glb`: PBRマテリアル付きGLBファイル
- `output.obj`: OBJメッシュ
- `run.log`: 実行ログ

## 技術スタック

- [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) (MIT License)
- CUDA 12.4
- PyTorch 2.6.0
- xformers (V100互換)

## License

MIT
