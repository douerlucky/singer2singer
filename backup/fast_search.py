import os
import json
import time
import random
import requests
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import heapq

# ----------------------------
# Config
# ----------------------------

NCM_BASE_URL = "http://localhost:3000"     # 你的网易云本地API
NCM_TIMEOUT = 15

MAX_DEPTH = 10                              # 超过就认为找不到
MAX_PAGES_PER_ARTIST = 5                  # 每个歌手最多翻几页（越大越慢）
PAGE_LIMIT = 100                            # 每页歌曲数量（网易云接口）
NEIGHBOR_TOPK = 12                          # LLM每次建议扩展的下一跳数量（越大越慢）
BEAM_WIDTH = 25                             # 每一层最多保留多少候选（越大越慢）
MAX_EXPANSIONS = 400                        # 最多展开多少个歌手节点，防止跑飞

# DeepSeek (OpenAI-compatible)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").strip()
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
DEEPSEEK_TIMEOUT = 20

# LLM调用失败时的降级策略：按“合作邻居数量”优先（近似枢纽优先）
FALLBACK_PICK_TOPK = 12


@dataclass(frozen=True)
class Artist:
    id: int
    name: str


@dataclass
class StepEdge:
    frm: Artist
    to: Artist
    songs: List[str]


# ----------------------------
# HTTP helpers
# ----------------------------

def ncm_get(path: str, params: dict) -> dict:
    r = requests.get(f"{NCM_BASE_URL}{path}", params=params, timeout=NCM_TIMEOUT)
    r.raise_for_status()
    return r.json()


def deepseek_chat_json(system: str, user: str, max_tokens: int = 350) -> dict:
    """
    调 DeepSeek OpenAI-compatible /chat/completions，要求返回JSON对象。
    """
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("未设置环境变量 DEEPSEEK_API_KEY")

    url = f"{DEEPSEEK_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
        # DeepSeek 支持 OpenAI 兼容字段；若服务端不支持也不会影响（多数会忽略）
        "response_format": {"type": "json_object"},
    }

    r = requests.post(url, headers=headers, json=payload, timeout=DEEPSEEK_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    return json.loads(content)


# ----------------------------
# NCM wrappers
# ----------------------------

def search_artist_first(name: str) -> Artist:
    data = ncm_get("/search", {"keywords": name, "type": 100})
    artists = data.get("result", {}).get("artists", [])
    if not artists:
        raise ValueError(f"没有搜索到歌手：{name}")
    a0 = artists[0]
    return Artist(id=int(a0["id"]), name=str(a0["name"]))


def fetch_collab_neighbors_fast(artist_id: int) -> Tuple[Dict[int, List[str]], Dict[int, str]]:
    """
    快速获取“合唱合作邻居”：
    - 只翻 MAX_PAGES_PER_ARTIST 页
    - 只提取 ar.length>1 的合唱
    返回：
      neighbors: neighbor_id -> [song_names]
      name_map:  artist_id -> artist_name (从ar里顺便学到)
    """
    neighbors: Dict[int, List[str]] = defaultdict(list)
    name_map: Dict[int, str] = {}

    offset = 0
    for _ in range(MAX_PAGES_PER_ARTIST):
        page = ncm_get("/artist/songs", {"id": artist_id, "limit": PAGE_LIMIT, "offset": offset})
        songs = page.get("songs", [])
        if not songs:
            break

        for song in songs:
            ars = song.get("ar", [])
            if not isinstance(ars, list) or len(ars) <= 1:
                continue

            sname = str(song.get("name", ""))

            for a in ars:
                try:
                    aid = int(a.get("id"))
                    an = str(a.get("name"))
                except Exception:
                    continue
                if aid and an:
                    name_map[aid] = an

            for a in ars:
                try:
                    nb = int(a.get("id"))
                except Exception:
                    continue
                if nb == artist_id:
                    continue
                neighbors[nb].append(sname)

        # 下一页
        if len(songs) < PAGE_LIMIT:
            break
        offset += PAGE_LIMIT

    # 去重歌曲名
    for nb, lst in list(neighbors.items()):
        seen = set()
        dedup = []
        for x in lst:
            if x in seen:
                continue
            seen.add(x)
            dedup.append(x)
        neighbors[nb] = dedup

    return dict(neighbors), name_map


# ----------------------------
# LLM ranking (heuristic)
# ----------------------------

def llm_rank_next_hops(cur: Artist, target: Artist, neighbors: Dict[int, List[str]], id2name: Dict[int, str]) -> List[int]:
    """
    给当前节点的邻居排序，返回一个 neighbor_id 列表（优先级从高到低）。
    为速度考虑：只把每个邻居的少量信息发给LLM。
    """
    # 构造候选摘要（只发前 2 首连接歌曲名）
    candidates = []
    for nb, songs in neighbors.items():
        nm = id2name.get(nb, str(nb))
        preview = songs[:2]
        candidates.append({
            "id": nb,
            "name": nm,
            "song_count": len(songs),
            "song_preview": preview
        })

    # 控制prompt大小：如果邻居太多，先按“song_count”筛一遍
    if len(candidates) > 80:
        candidates.sort(key=lambda x: x["song_count"], reverse=True)
        candidates = candidates[:80]

    system = (
        "你是一个图搜索启发式专家。我们在“歌手合作图”里找从start到target的一条可行路径（不要求最短）。"
        "给定当前歌手cur与target，以及cur的合作邻居列表。请挑选最可能通向target圈层的下一跳，返回JSON。"
        "只返回JSON，不要输出多余文字。"
    )
    user = {
        "task": "rank_next_hops",
        "cur": {"id": cur.id, "name": cur.name},
        "target": {"id": target.id, "name": target.name},
        "candidates": candidates,
        "output_requirements": {
            "topk": NEIGHBOR_TOPK,
            "format": {"next_ids": "list[int] ordered high->low"}
        }
    }

    try:
        out = deepseek_chat_json(system, json.dumps(user, ensure_ascii=False))
        next_ids = out.get("next_ids", [])
        # 清洗：只保留真实邻居 & 去重
        seen = set()
        ranked = []
        neighbor_set = set(neighbors.keys())
        for x in next_ids:
            try:
                xid = int(x)
            except Exception:
                continue
            if xid in neighbor_set and xid not in seen:
                seen.add(xid)
                ranked.append(xid)
        if ranked:
            return ranked
    except Exception:
        pass

    # fallback：按合作歌曲数量多的优先（枢纽歌手更容易连通）
    items = sorted(neighbors.items(), key=lambda kv: len(kv[1]), reverse=True)
    return [nb for nb, _ in items[:FALLBACK_PICK_TOPK]]


# ----------------------------
# Fast search (beam / best-first)
# ----------------------------

class LLMFastPathFinder:
    def __init__(self):
        self.artist_cache: Dict[int, Artist] = {}
        self.nei_cache: Dict[int, Dict[int, List[str]]] = {}
        self.name_cache: Dict[int, str] = {}          # id->name（从歌曲ar里学）
        self.edge_songs: Dict[Tuple[int, int], List[str]] = {}  # (u,v)->songs

    def get_artist(self, aid: int) -> Artist:
        if aid in self.artist_cache:
            return self.artist_cache[aid]
        nm = self.name_cache.get(aid, str(aid))
        a = Artist(aid, nm)
        self.artist_cache[aid] = a
        return a

    def neighbors(self, aid: int) -> Dict[int, List[str]]:
        if aid in self.nei_cache:
            return self.nei_cache[aid]
        neighbors, name_map = fetch_collab_neighbors_fast(aid)
        self.name_cache.update(name_map)

        # 缓存边上的歌曲
        for nb, songs in neighbors.items():
            self.edge_songs[(aid, nb)] = songs

        self.nei_cache[aid] = neighbors
        return neighbors

    def reconstruct(self, parent: Dict[int, int], start: int, end: int) -> List[StepEdge]:
        chain = [end]
        cur = end
        while cur != start:
            cur = parent[cur]
            chain.append(cur)
        chain.reverse()

        steps: List[StepEdge] = []
        for i in range(len(chain) - 1):
            u, v = chain[i], chain[i + 1]
            frm = self.get_artist(u)
            to = self.get_artist(v)
            songs = self.edge_songs.get((u, v), [])
            steps.append(StepEdge(frm=frm, to=to, songs=songs))
        return steps

    def find_path(self, start: Artist, target: Artist) -> Optional[List[StepEdge]]:
        self.artist_cache[start.id] = start
        self.artist_cache[target.id] = target
        self.name_cache[start.id] = start.name
        self.name_cache[target.id] = target.name

        # best-first: 用一个小根堆，优先扩展“更有希望”的节点
        # 这里 score = depth（基础） + small_noise（打散）
        heap: List[Tuple[float, int]] = []
        heapq.heappush(heap, (0.0, start.id))

        parent: Dict[int, int] = {}
        depth: Dict[int, int] = {start.id: 0}
        visited: set[int] = {start.id}

        expansions = 0

        while heap and expansions < MAX_EXPANSIONS:
            _, cur_id = heapq.heappop(heap)
            d = depth[cur_id]
            if d >= MAX_DEPTH:
                continue
            if cur_id == target.id:
                return self.reconstruct(parent, start.id, target.id)

            cur_artist = self.get_artist(cur_id)

            # 拿邻居（快版：最多翻 MAX_PAGES_PER_ARTIST 页）
            nbrs = self.neighbors(cur_id)
            if not nbrs:
                continue

            # 让 LLM 给下一跳排序
            ranked_ids = llm_rank_next_hops(cur_artist, target, nbrs, self.name_cache)

            # Beam：只扩展前 BEAM_WIDTH 个（更狠一点）
            ranked_ids = ranked_ids[:BEAM_WIDTH]

            for nb in ranked_ids:
                if nb in visited:
                    continue
                visited.add(nb)
                parent[nb] = cur_id
                depth[nb] = d + 1

                # 保存从 cur->nb 的 songs（供展示）
                if (cur_id, nb) not in self.edge_songs:
                    self.edge_songs[(cur_id, nb)] = nbrs.get(nb, [])

                if nb == target.id:
                    return self.reconstruct(parent, start.id, target.id)

                # score：更小优先。这里用 depth + 一个很小的噪声，避免完全一致导致路径单一
                score = (d + 1) + random.random() * 0.01
                heapq.heappush(heap, (score, nb))

            expansions += 1

        return None


# ----------------------------
# Demo entry
# ----------------------------

if __name__ == "__main__":
    start_name = input("Start singer name: ").strip()
    end_name = input("End singer name: ").strip()

    start = search_artist_first(start_name)
    end = search_artist_first(end_name)

    print(f"Start: {start.name} ({start.id})")
    print(f"End:   {end.name} ({end.id})")
    print(f"Config: MAX_DEPTH={MAX_DEPTH}, MAX_PAGES_PER_ARTIST={MAX_PAGES_PER_ARTIST}, TOPK={NEIGHBOR_TOPK}, BEAM={BEAM_WIDTH}")

    finder = LLMFastPathFinder()

    t0 = time.time()
    path = finder.find_path(start, end)
    t1 = time.time()

    if not path:
        print(f"\n在深度 <= {MAX_DEPTH} 内没找到可行路径（用时 {t1 - t0:.2f}s）。")
    else:
        print(f"\n找到一条路径：{len(path)} 步（用时 {t1 - t0:.2f}s）")
        for i, step in enumerate(path, 1):
            songs_preview = step.songs[:5]
            more = " ..." if len(step.songs) > 5 else ""
            print(f"{i}. {step.frm.name} -> {step.to.name} via {songs_preview}{more}")