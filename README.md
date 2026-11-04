# DoG_AI_Back

DoG_AI の仮バックエンドです。フロントエンド接続確認用に、最小限の API を提供します。

## 起動方法

```bash
npm install
npm start
```

デフォルトでは `http://localhost:3000` で起動します。

## 仮 API

- `GET /health` : ヘルスチェック
- `GET /api/dogs` : 仮の犬データ一覧
- `POST /api/dogs/analyze` : 仮の解析結果を返す

## テスト

```bash
npm test
```
