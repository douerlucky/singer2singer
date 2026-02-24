import requests
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import deque

BASE_URL = "http://localhost:3000"
MAX_DEPTH = 10 #BFS递归深度
PAGE_LIMIT = 100 #每一页获取的最多歌曲数
TIMEOUT = 15   #超时

MAX_SONGS_PER_ARTIST = 600  #每个歌手最多搜索的歌曲


@dataclass(frozen=True)
class Artist:
    id: int
    name: str
    pic_url: str = ""


@dataclass(frozen=True)
class SongBrief:
    id: int
    name: str
    cover: str
    artists: tuple = ()   # 歌手元组


@dataclass
class StepEdge:
    frm: Artist
    song: SongBrief
    to: Artist



def _get(path: str, params: dict) -> dict:
    r = requests.get(f"{BASE_URL}{path}", params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()



def search_artist_first(name: str) -> Artist:
    data = _get("/search", {"keywords": name, "type": 100})
    artists = data.get("result", {}).get("artists", [])
    if not artists:
        raise ValueError(f"没有搜索到歌手：{name}")
    a0 = artists[0]
    return Artist(id=int(a0["id"]), name=str(a0["name"]), pic_url=str(a0.get("picUrl") or ""))


def fetch_songs_limited(artist_id: int, max_songs: int = MAX_SONGS_PER_ARTIST) -> List[dict]:
    all_songs: List[dict] = []
    offset = 0
    while len(all_songs) < max_songs:
        page = _get("/artist/songs", {"id": artist_id, "limit": PAGE_LIMIT, "offset": offset})
        songs = page.get("songs", [])
        if not songs:
            break
        need = max_songs - len(all_songs)
        all_songs.extend(songs[:need])
        if len(songs) < PAGE_LIMIT:
            break
        offset += PAGE_LIMIT
    seen = set()
    uniq = []
    for s in all_songs:
        sid = s.get("id")
        if sid in seen:
            continue
        seen.add(sid)
        uniq.append(s)
    return uniq


def fetch_song_cover(song_id: int) -> str:
    try:
        data = _get("/song/detail", {"ids": song_id})
        songs = data.get("songs") or []
        if not songs:
            return ""
        al = songs[0].get("al") or {}
        return str(al.get("picUrl") or "")
    except Exception:
        return ""


def extract_collab_edges_one_song(song_list: List[dict], focus_artist_id: int) -> Dict[int, SongBrief]:
    edges: Dict[int, SongBrief] = {}

    for song in song_list:
        ars = song.get("ar", [])
        if not isinstance(ars, list) or len(ars) <= 1:
            continue

        try:
            sid = int(song.get("id"))
        except Exception:
            continue

        sname = str(song.get("name", ""))
        cover = fetch_song_cover(sid)

        # Collect all artist names for this song
        artist_names = tuple(str(a.get("name", "")) for a in ars if a.get("name"))

        for a in ars:
            try:
                aid = int(a.get("id"))
            except Exception:
                continue
            if aid == focus_artist_id:
                continue

            if aid not in edges:
                edges[aid] = SongBrief(id=sid, name=sname, cover=cover, artists=artist_names)

    return edges


def fetch_artist_pic(artist_id: int) -> str:
    try:
        data = _get("/artists", {"id": artist_id})
        artist = data.get("artist") or {}
        return str(artist.get("picUrl") or "")
    except Exception:
        return ""


#BFS最短路径
class CollabGraphSearcher:
    def __init__(self):
        self._artist_cache: Dict[int, Artist] = {}
        self._neighbors_cache: Dict[int, Dict[int, SongBrief]] = {}

    def _ensure_artist_cached(self, artist_id: int, fallback_name: Optional[str] = None):
        if artist_id in self._artist_cache:
            return
        pic = fetch_artist_pic(artist_id)
        self._artist_cache[artist_id] = Artist(
            id=artist_id,
            name=fallback_name or str(artist_id),
            pic_url=pic
        )

    def neighbors(self, artist_id: int) -> Dict[int, SongBrief]:
        if artist_id in self._neighbors_cache:
            return self._neighbors_cache[artist_id]

        songs = fetch_songs_limited(artist_id)

        if artist_id not in self._artist_cache:
            me_name = None
            for s in songs:
                for a in s.get("ar", []):
                    try:
                        if int(a.get("id", -1)) == artist_id:
                            me_name = str(a.get("name"))
                            break
                    except Exception:
                        pass
                if me_name:
                    break
            self._ensure_artist_cached(artist_id, fallback_name=me_name)

        for s in songs:
            ars = s.get("ar", [])
            if not isinstance(ars, list):
                continue
            for a in ars:
                try:
                    aid = int(a.get("id"))
                    aname = str(a.get("name") or "")
                except Exception:
                    continue
                if aid not in self._artist_cache:
                    self._artist_cache[aid] = Artist(id=aid, name=aname, pic_url="")

        edges = extract_collab_edges_one_song(songs, artist_id)
        self._neighbors_cache[artist_id] = edges
        return edges

    def shortest_path(self, start: Artist, end: Artist, max_depth: int = MAX_DEPTH) -> Optional[List[StepEdge]]:
        if not start.pic_url:
            start = Artist(start.id, start.name, fetch_artist_pic(start.id))
        if not end.pic_url:
            end = Artist(end.id, end.name, fetch_artist_pic(end.id))

        self._artist_cache[start.id] = start
        self._artist_cache[end.id] = end

        q = deque([start.id])
        visited = {start.id}
        parent: Dict[int, int] = {}
        edge_song: Dict[int, SongBrief] = {}
        depth: Dict[int, int] = {start.id: 0}

        while q:
            cur = q.popleft()
            d = depth[cur]
            if d >= max_depth:
                continue
            if cur == end.id:
                break

            nbrs = self.neighbors(cur)
            for nb, song in nbrs.items():
                if nb in visited:
                    continue
                visited.add(nb)
                parent[nb] = cur
                edge_song[nb] = song
                depth[nb] = d + 1

                if nb == end.id:
                    q.clear()
                    break
                q.append(nb)

        if end.id not in visited:
            return None

        chain_ids = [end.id]
        cur = end.id
        while cur != start.id:
            cur = parent[cur]
            chain_ids.append(cur)
        chain_ids.reverse()

        steps: List[StepEdge] = []
        for i in range(len(chain_ids) - 1):
            a = chain_ids[i]
            b = chain_ids[i + 1]
            frm = self._artist_cache.get(a, Artist(a, str(a), ""))
            to = self._artist_cache.get(b, Artist(b, str(b), ""))

            if not frm.pic_url:
                frm = Artist(frm.id, frm.name, fetch_artist_pic(frm.id))
                self._artist_cache[frm.id] = frm
            if not to.pic_url:
                to = Artist(to.id, to.name, fetch_artist_pic(to.id))
                self._artist_cache[to.id] = to

            song = edge_song[b]
            steps.append(StepEdge(frm=frm, song=song, to=to))
        return steps


if __name__ == "__main__":
    start_name = input("Start singer name: ").strip()
    end_name = input("End singer name: ").strip()

    start = search_artist_first(start_name)
    end = search_artist_first(end_name)

    searcher = CollabGraphSearcher()
    path = searcher.shortest_path(start, end, max_depth=MAX_DEPTH)

    if not path:
        print(f"\n在深度 <= {MAX_DEPTH} 内找不到路径。")
    else:
        print(f"\n找到路径：{len(path)} 步")
        for idx, step in enumerate(path, 1):
            print(f"{idx}. {step.frm.name} -> {step.song.name} -> {step.to.name}")