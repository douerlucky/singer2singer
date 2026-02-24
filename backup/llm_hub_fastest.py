import os
import json
import time
import random
import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# =========================
# Config (越快越小)
# =========================
NCM_BASE_URL = "http://localhost:3000"
NCM_TIMEOUT = 12

MAX_DEPTH = 10                 # 深度上限（步数）
BEAM_WIDTH = 10                # 每层保留多少候选（越小越快，建议 6~15）
NEXT_TOPK = 6                  # 每次从当前点扩展多少下一跳（越小越快，建议 4~10）
MAX_EXPANSIONS = 160           # 总扩展节点上限（防跑飞）

# 每个歌手用于“生成邻居”的分页页数（极速版默认只取第1页）
PAGES_FOR_NEIGHBORS = 1
PAGE_LIMIT = 100

# 为了评估“枢纽度”，对候选邻居抓第1页估计 degree（并发）
DEGREE_ESTIMATE_PAGES = 1
DEGREE_WORKERS = 10            # 并发线程数（本地API一般可以大一点）

# DeepSeek (OpenAI-compatible)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").strip()
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
DEEPSEEK_TIMEOUT = 20

# LLM 调用频率控制
LLM_EVERY_N_STEPS = 1          # 每一步都用LLM（更准但更慢），可改为 2/3
LLM_CANDIDATE_CAP = 25         # 发给LLM的候选最多多少（越小越快）


@dataclass(frozen=True)
class Artist:
    id: int
    name: str


@dataclass
class StepEdge:
    frm: Artist
    to: Artist
    songs: List[str]


# =========================
# HTTP: reuse session
# =========================
ncm_sess = requests.Session()
ds_sess = requests.Session()


def ncm_get(path: str, params: dict) -> dict:
    r = ncm_sess.get(f"{NCM_BASE_URL}{path}", params=params, timeout=NCM_TIMEOUT)
    r.raise_for_status()
    return r.json()


def deepseek_chat_json(system: str, user_obj: dict, max_tokens: int = 280) -> dict:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("未设置环境变量 DEEPSEEK_API_KEY")

    url = f"{DEEPSEEK_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_obj, ensure_ascii=False)},
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    r = ds_sess.post(url, headers=headers, json=payload, timeout=DEEPSEEK_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    return json.loads(content)


# =========================
# NCM wrappers
# =========================
def search_artist_first(name: str) -> Artist:
    data = ncm_get("/search", {"keywords": name, "type": 100})
    artists = data.get("result", {}).get("artists", [])
    if not artists:
        raise ValueError(f"没有搜索到歌手：{name}")
    a0 = artists[0]
    return Artist(id=int(a0["id"]), name=str(a0["name"]))


def fetch_collab_neighbors(
    artist_id: int,
    pages: int,
    limit: int = PAGE_LIMIT
) -> Tuple[Dict[int, List[str]], Dict[int, str]]:
    """
    返回：
      neighbors: nb_id -> [song_name,...] (合唱才算)
      name_map:  id -> name（从 ar 里学到）
    """
    neighbors: Dict[int, List[str]] = defaultdict(list)
    name_map: Dict[int, str] = {}
    offset = 0

    for _ in range(pages):
        page = ncm_get("/artist/songs", {"id": artist_id, "limit": limit, "offset": offset})
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

        if len(songs) < limit:
            break
        offset += limit

    # dedup song names
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


def estimate_degree_quick(artist_id: int) -> int:
    """
    用极低成本估计一个歌手“枢纽度”：抓 1 页合唱，统计 unique 合作歌手数。
    """
    try:
        nbrs, _ = fetch_collab_neighbors(artist_id, pages=DEGREE_ESTIMATE_PAGES)
        return len(nbrs)
    except Exception:
        return 0


# =========================
# LLM: choose promising next hops
# =========================
def llm_pick_next_ids(
    cur: Artist,
    target: Artist,
    candidates: List[dict],
    topk: int
) -> List[int]:
    """
    candidates: [{id,name,degree,song_preview,song_count}, ...]
    返回 next_ids（按优先级）
    """
    system = (
        "你是一个路径搜索助手。我们在“歌手合作网络”里，从 cur 走到 target，"
        "只需要尽快找到一条可行路径（不保证最短）。"
        "请优先选择：更可能接近 target 圈层 + 连接度(degree)更高的枢纽歌手。"
        "只返回 JSON。"
    )
    user = {
        "task": "pick_next_hops",
        "cur": {"id": cur.id, "name": cur.name},
        "target": {"id": target.id, "name": target.name},
        "candidates": candidates,
        "topk": topk,
        "output": {"next_ids": "list[int]"}
    }

    out = deepseek_chat_json(system, user, max_tokens=220)
    next_ids = out.get("next_ids", [])
    cleaned = []
    seen = set()
    cand_set = {c["id"] for c in candidates}
    for x in next_ids:
        try:
            xid = int(x)
        except Exception:
            continue
        if xid in cand_set and xid not in seen:
            seen.add(xid)
            cleaned.append(xid)
    return cleaned


# =========================
# Fastest path finder
# =========================
class HubLLMFastest:
    def __init__(self):
        self.name_cache: Dict[int, str] = {}
        self.nei_cache: Dict[int, Dict[int, List[str]]] = {}
        self.edge_songs: Dict[Tuple[int, int], List[str]] = {}

    def get_name(self, aid: int) -> str:
        return self.name_cache.get(aid, str(aid))

    def get_artist(self, aid: int) -> Artist:
        return Artist(aid, self.get_name(aid))

    def neighbors_fast(self, aid: int) -> Dict[int, List[str]]:
        if aid in self.nei_cache:
            return self.nei_cache[aid]
        nbrs, name_map = fetch_collab_neighbors(aid, pages=PAGES_FOR_NEIGHBORS)
        self.name_cache.update(name_map)
        for nb, songs in nbrs.items():
            self.edge_songs[(aid, nb)] = songs
        self.nei_cache[aid] = nbrs
        return nbrs

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
            steps.append(
                StepEdge(
                    frm=self.get_artist(u),
                    to=self.get_artist(v),
                    songs=self.edge_songs.get((u, v), [])
                )
            )
        return steps

    def find_fastest_path(self, start: Artist, target: Artist) -> Optional[List[StepEdge]]:
        self.name_cache[start.id] = start.name
        self.name_cache[target.id] = target.name

        # Beam state: (score, node_id)
        beam = [(0.0, start.id)]
        parent: Dict[int, int] = {}
        depth: Dict[int, int] = {start.id: 0}
        visited: set[int] = {start.id}

        expansions = 0
        step_no = 0

        while beam and expansions < MAX_EXPANSIONS:
            # 取当前最优的几个（beam）
            beam.sort(key=lambda x: x[0])
            beam = beam[:BEAM_WIDTH]

            next_beam = []
            step_no += 1

            for score, cur_id in beam:
                d = depth[cur_id]
                if d >= MAX_DEPTH:
                    continue
                if cur_id == target.id:
                    return self.reconstruct(parent, start.id, target.id)

                cur = self.get_artist(cur_id)
                nbrs = self.neighbors_fast(cur_id)
                if not nbrs:
                    continue

                # 先把候选按“本边的合唱数”粗排（局部信号）
                items = sorted(nbrs.items(), key=lambda kv: len(kv[1]), reverse=True)

                # 只取一小撮候选做后续（很关键：别把几百个邻居都处理）
                items = items[:LLM_CANDIDATE_CAP]
                cand_ids = [nb for nb, _ in items]

                # 并发估计这些候选的 degree（枢纽度）
                deg_map: Dict[int, int] = {}
                with ThreadPoolExecutor(max_workers=DEGREE_WORKERS) as ex:
                    futs = {ex.submit(estimate_degree_quick, nb): nb for nb in cand_ids}
                    for fut in as_completed(futs):
                        nb = futs[fut]
                        deg_map[nb] = fut.result()

                # 组候选给 LLM：id/name/degree/歌曲线索
                candidates = []
                for nb, songs in items:
                    candidates.append({
                        "id": nb,
                        "name": self.get_name(nb),
                        "degree": deg_map.get(nb, 0),
                        "song_count_with_cur": len(songs),
                        "song_preview": songs[:2],
                    })

                # LLM 选下一跳（可调：不是每一步都用）
                use_llm = (step_no % LLM_EVERY_N_STEPS == 0) and DEEPSEEK_API_KEY
                if use_llm:
                    try:
                        picked = llm_pick_next_ids(cur, target, candidates, topk=NEXT_TOPK)
                    except Exception:
                        picked = []
                else:
                    picked = []

                # 如果 LLM 没给出结果，fallback：按 (degree, song_count_with_cur) 排
                if not picked:
                    candidates.sort(key=lambda c: (c["degree"], c["song_count_with_cur"]), reverse=True)
                    picked = [c["id"] for c in candidates[:NEXT_TOPK]]

                # 扩展 picked
                for nb in picked:
                    if nb in visited:
                        continue
                    visited.add(nb)
                    parent[nb] = cur_id
                    depth[nb] = d + 1

                    # 保存边歌名
                    if (cur_id, nb) not in self.edge_songs:
                        self.edge_songs[(cur_id, nb)] = nbrs.get(nb, [])

                    if nb == target.id:
                        return self.reconstruct(parent, start.id, target.id)

                    # score 越小越优先：深度优先 + 奖励枢纽（degree越大，score越小）
                    deg = deg_map.get(nb, 0)
                    new_score = (d + 1) - min(deg, 200) / 200.0 + random.random() * 0.01
                    next_beam.append((new_score, nb))

                expansions += 1
                if expansions >= MAX_EXPANSIONS:
                    break

            beam = next_beam

        return None


# =========================
# Demo
# =========================
if __name__ == "__main__":
    start_name = input("Start singer name: ").strip()
    end_name = input("End singer name: ").strip()

    start = search_artist_first(start_name)
    end = search_artist_first(end_name)

    print(f"Start: {start.name} ({start.id})")
    print(f"End:   {end.name} ({end.id})")
    print(f"Config: MAX_DEPTH={MAX_DEPTH}, BEAM_WIDTH={BEAM_WIDTH}, NEXT_TOPK={NEXT_TOPK}, "
          f"PAGES_FOR_NEIGHBORS={PAGES_FOR_NEIGHBORS}, DEG_WORKERS={DEGREE_WORKERS}")

    finder = HubLLMFastest()

    t0 = time.time()
    path = finder.find_fastest_path(start, end)
    t1 = time.time()

    if not path:
        print(f"\n没找到（深度<= {MAX_DEPTH}）。用时 {t1 - t0:.2f}s")
    else:
        print(f"\n找到一条路径：{len(path)} 步，用时 {t1 - t0:.2f}s")
        for i, step in enumerate(path, 1):
            songs_preview = step.songs[:4]
            more = " ..." if len(step.songs) > 4 else ""
            print(f"{i}. {step.frm.name} -> {step.to.name} via {songs_preview}{more}")