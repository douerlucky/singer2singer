from flask import Flask, render_template, request, jsonify
from main_moreworker import search_artist_first, CollabGraphSearcher
import time
from flask import Response
import requests

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/img-proxy")
def img_proxy():
    url = request.args.get("url")
    if not url:
        return "No url", 400
    try:
        headers = {
            "Referer": "https://music.163.com/",
            "User-Agent": "Mozilla/5.0"
        }
        r = requests.get(url, headers=headers, timeout=10)
        return Response(r.content, content_type=r.headers.get("Content-Type"))
    except Exception:
        return "Error", 500


@app.route("/search", methods=["POST"])
def search():
    data = request.json or {}
    start_name = (data.get("start") or "").strip()
    end_name = (data.get("end") or "").strip()
    if not start_name or not end_name:
        return jsonify({"success": False, "message": "请输入开始/结束歌手名"})

    try:
        t0 = time.time()
        start = search_artist_first(start_name)
        end = search_artist_first(end_name)

        searcher = CollabGraphSearcher()
        path = searcher.shortest_path(start, end)

        if not path:
            return jsonify({"success": False, "message": "未找到路径（可尝试换关键词或减小范围）"})

        result = []
        for step in path:
            result.append({
                "from": {
                    "name": step.frm.name,
                    "id": step.frm.id,
                    "picUrl": step.frm.pic_url,
                    "url": f"https://music.163.com/#/artist?id={step.frm.id}"
                },
                "song": {
                    "name": step.song.name,
                    "id": step.song.id,
                    "cover": step.song.cover,
                    "artists": list(step.song.artists),
                    "url": f"https://music.163.com/#/song?id={step.song.id}"
                },
                "to": {
                    "name": step.to.name,
                    "id": step.to.id,
                    "picUrl": step.to.pic_url,
                    "url": f"https://music.163.com/#/artist?id={step.to.id}"
                }
            })

        return jsonify({
            "success": True,
            "start": start.name,
            "end": end.name,
            "steps": len(result),
            "elapsed": round(time.time() - t0, 2),
            "path": result
        })

    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


if __name__ == "__main__":
    app.run(debug=True)