import threading
import requests
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

BASE_URL = "http://localhost:3000"
MAX_DEPTH = 10 #BFS递归深度
PAGE_LIMIT = 100 #每一页获取的最多歌曲数
TIMEOUT = 15   #超时
MAX_SONGS_PER_ARTIST = 600  #每个歌手最多搜索的歌曲
MAX_WORKERS = 8 #线程数

# 每个线程独立持有一个 Session
_thread_local = threading.local()


def _get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        # 降低连接池大小，防止连接泄漏
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=2,
            pool_maxsize=2,
            max_retries=0,
        )
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        _thread_local.session = s
    return _thread_local.session


@dataclass(frozen=True)
class Artist:
    id: int
    name: str
    pic_url: str = ""


@dataclass(frozen=True)
class SongBrief:
    id: int
    name: str
    cover: str = ""
    artists: tuple = ()


@dataclass
class StepEdge:
    frm: Artist
    song: SongBrief
    to: Artist


# ----------------------------
def _get(path: str, params: dict) -> dict:
    session = _get_session()
    max_retries = 2
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            r = session.get(
                f"{BASE_URL}{path}",
                params=params,
                timeout=TIMEOUT
            )
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            raise ValueError(f"API 请求超时 ({TIMEOUT}s)：{path}")
        except requests.exceptions.ConnectionError as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            raise ValueError(
                f"无法连接到 API 服务器 {BASE_URL}\n"
                f"检查：pm2 logs netease-api"
            )
        except Exception as e:
            raise ValueError(f"API 错误：{str(e)}")

    raise ValueError("请求失败")


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

    # 去重
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


def fetch_artist_pic(artist_id: int) -> str:
    try:
        data = _get("/artists", {"id": artist_id})
        artist = data.get("artist") or {}
        return str(artist.get("picUrl") or "")
    except Exception:
        return ""



def extract_collab_edges_no_cover(song_list: List[dict], focus_artist_id: int) -> Dict[int, SongBrief]:
    #BFS只建边
    edges: Dict[int, SongBrief] = {}

    for song in song_list:
        ars = song.get("ar", [])
        if not isinstance(ars, list) or len(ars) <= 1:
            continue
        try:
            sid = int(song["id"])
        except Exception:
            continue

        sname = str(song.get("name", ""))
        artist_names = tuple(str(a.get("name", "")) for a in ars if a.get("name"))

        for a in ars:
            try:
                aid = int(a["id"])
            except Exception:
                continue
            if aid == focus_artist_id or aid in edges:
                continue
            edges[aid] = SongBrief(id=sid, name=sname, cover="", artists=artist_names)

    return edges


def fill_covers_for_path(songs: List[SongBrief]) -> Dict[int, str]:
    #路径确定后，只对路径上的歌曲并行抓封面
    if not songs:
        return {}
    covers: Dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(songs))) as ex:
        future_to_id = {ex.submit(fetch_song_cover, s.id): s.id for s in songs}
        for f in as_completed(future_to_id):
            sid = future_to_id[f]
            try:
                covers[sid] = f.result()
            except Exception:
                covers[sid] = ""
    return covers


class CollabGraphSearcher:

    def __init__(self, progress_callback: Optional[Callable[[str], None]] = None,
                 stop_event: Optional[threading.Event] = None):
        # progress_callback进度回调函数
        # stop_event停止标志，立即停止搜索

        self._artist_cache: Dict[int, Artist] = {}
        self._neighbors_cache: Dict[int, Dict[int, SongBrief]] = {}
        self._cache_lock = threading.Lock()
        self._progress = progress_callback if callable(progress_callback) else (lambda msg: None)
        self._stop_event = stop_event

    def _progress_msg(self, msg: str):
        # 检查是否需要停止
        if self._stop_event and self._stop_event.is_set():
            raise RuntimeError("搜索已停止")
        try:
            self._progress(msg)
        except Exception:
            pass

    def _cache_artists_from_songs(self, songs: List[dict]) -> None:
        with self._cache_lock:
            for s in songs:
                for a in s.get("ar", []):
                    try:
                        aid = int(a["id"])
                        aname = str(a.get("name") or "")
                    except Exception:
                        continue
                    if aid not in self._artist_cache:
                        self._artist_cache[aid] = Artist(id=aid, name=aname, pic_url="")

    def neighbors(self, artist_id: int) -> Dict[int, SongBrief]:
        #获取艺术家的合作者（BFS 展开）
        if self._stop_event and self._stop_event.is_set():
            raise RuntimeError("搜索已停止")

        with self._cache_lock:
            if artist_id in self._neighbors_cache:
                return self._neighbors_cache[artist_id]

        with self._cache_lock:
            artist_name = self._artist_cache.get(artist_id)
        name_str = artist_name.name if artist_name else str(artist_id)
        self._progress_msg(f"🎵 正在获取「{name_str}」的歌曲…")

        songs = fetch_songs_limited(artist_id)
        self._cache_artists_from_songs(songs)

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

        with self._cache_lock:
            if artist_id not in self._artist_cache:
                self._artist_cache[artist_id] = Artist(
                    id=artist_id,
                    name=me_name or str(artist_id),
                    pic_url=""
                )

        edges = extract_collab_edges_no_cover(songs, artist_id)

        with self._cache_lock:
            if artist_id not in self._neighbors_cache:
                self._neighbors_cache[artist_id] = edges
            return self._neighbors_cache[artist_id]

    def _fill_pic(self, artist: Artist) -> Artist:
        if artist.pic_url:
            return artist
        pic = fetch_artist_pic(artist.id)
        updated = Artist(artist.id, artist.name, pic)
        with self._cache_lock:
            self._artist_cache[artist.id] = updated
        return updated

    def shortest_path(self, start: Artist, end: Artist, max_depth: int = MAX_DEPTH) -> Optional[List[StepEdge]]:
        #BFS找最短路径
        start = self._fill_pic(start)
        end = self._fill_pic(end)

        with self._cache_lock:
            self._artist_cache[start.id] = start
            self._artist_cache[end.id] = end

        if start.id == end.id:
            return []

        q: deque = deque([start.id])
        visited: set = {start.id}
        parent: Dict[int, int] = {}
        edge_song: Dict[int, SongBrief] = {}
        depth: Dict[int, int] = {start.id: 0}

        found = False
        bfs_lock = threading.Lock()

        while q and not found:
            # 检查停止标志
            if self._stop_event and self._stop_event.is_set():
                raise RuntimeError("搜索已停止")

            current_layer = []
            while q:
                current_layer.append(q.popleft())

            current_layer = [n for n in current_layer if depth[n] < max_depth]
            if not current_layer:
                break

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                future_map = {ex.submit(self.neighbors, cur): cur for cur in current_layer}

                for f in as_completed(future_map):
                    if found or (self._stop_event and self._stop_event.is_set()):
                        break
                    cur = future_map[f]
                    try:
                        nbrs = f.result()
                    except Exception:
                        continue

                    with bfs_lock:
                        if found:
                            break
                        for nb, song in nbrs.items():
                            if nb in visited:
                                continue
                            visited.add(nb)
                            parent[nb] = cur
                            edge_song[nb] = song
                            depth[nb] = depth[cur] + 1

                            if nb == end.id:
                                found = True
                                break

                            q.append(nb)

        if end.id not in parent:
            return None

        # 回溯路径
        chain_ids = [end.id]
        cur = end.id
        while cur != start.id:
            cur = parent[cur]
            chain_ids.append(cur)
        chain_ids.reverse()

        #路径上的歌曲和艺术家信息
        path_songs = [edge_song[chain_ids[i + 1]] for i in range(len(chain_ids) - 1)]
        covers = fill_covers_for_path(path_songs)

        path_artists_raw = []
        with self._cache_lock:
            for aid in set(chain_ids):
                path_artists_raw.append(self._artist_cache.get(aid, Artist(aid, str(aid), "")))

        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(path_artists_raw))) as ex:
            futures = [ex.submit(self._fill_pic, a) for a in path_artists_raw]
            for f in as_completed(futures):
                f.result()

        # 组装结果
        steps: List[StepEdge] = []
        for i in range(len(chain_ids) - 1):
            a, b = chain_ids[i], chain_ids[i + 1]
            with self._cache_lock:
                frm = self._artist_cache.get(a, Artist(a, str(a), ""))
                to = self._artist_cache.get(b, Artist(b, str(b), ""))
            raw_song = edge_song[b]
            song_with_cover = SongBrief(
                id=raw_song.id,
                name=raw_song.name,
                cover=covers.get(raw_song.id, ""),
                artists=raw_song.artists,
            )
            steps.append(StepEdge(frm=frm, song=song_with_cover, to=to))

        return steps

if __name__ == "__main__":
    start_name = input("Start singer name: ").strip()
    end_name = input("End singer name: ").strip()

    start = search_artist_first(start_name)
    end = search_artist_first(end_name)


    def my_progress(msg: str):
        print(f"[进度] {msg}")


    searcher = CollabGraphSearcher(progress_callback=my_progress)
    path = searcher.shortest_path(start, end, max_depth=MAX_DEPTH)

    if path is None:
        print(f"\n在深度 <= {MAX_DEPTH} 内找不到路径。")
    else:
        print(f"\n找到路径：{len(path)} 步")
        for idx, step in enumerate(path, 1):
            print(f"{idx}. {step.frm.name} →[{step.song.name}]→ {step.to.name}")