# AutoRepo

# イメージ削除
PS C:\Work\AutoRepo> docker rmi -f 3299fa737f2f
Untagged: autorepo:latest
Deleted: sha256:3299fa737f2f562dfd8502bdf8ec31a7962384a696dd1b355ee446ffa89a53b8

# ビルド
PS C:\Work\AutoRepo> docker image build -t autorepo .

# 実行
PS C:\Work\AutoRepo> docker run -p 8501:8501 autorepo


画像サイズ
https://juu7g.hatenablog.com/entry/Python/image/resize-sig
