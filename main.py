"""视频之眼：把视频抽帧拼成一张图，交给视觉模型看；同时把声音转成文字。

两种触发：
  1) 自动：私聊里发视频 / 发视频站点链接，直接处理（仅 snowluma_qq2 通道）
  2) 命令：/video <路径或链接>

听觉部分用本地 faster-whisper（CPU，int8）。模型和缓存都放 D 盘，
走 hf-mirror 下载，避免直连 HuggingFace 卡住。
"""

from __future__ import annotations

# 注意：这两个环境变量必须在 import faster_whisper 之前设置，
# 否则会去连 HuggingFace 主站（国内基本拉不动）。
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def _resolve_data_dir() -> Path:
    """AstrBot 的插件数据目录；拿不到就退回到插件自己的 data/。"""
    try:
        from astrbot.core.star.star_tools import StarTools

        d = StarTools.get_data_dir("astrbot_plugin_video_eyes")
        if d:
            return Path(d)
    except Exception:  # noqa: BLE001
        pass
    return Path(__file__).resolve().parent / "data"


DATA_DIR = _resolve_data_dir()
os.environ.setdefault("HF_HOME", str(DATA_DIR / "hf"))
# huggingface_hub 新版默认走 xet 传输，镜像站不支持，会 401；禁掉后走普通 HTTP。
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import asyncio
import datetime as _dt
import hashlib
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

import cv2
import httpx
import numpy as np

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

PLUGIN_NAME = "astrbot_plugin_video_eyes"
TEMP_ROOT = DATA_DIR / "frames"
DEBUG_LOG = DATA_DIR / "debug.log"
DEFAULT_SCOPE = ""
MAX_BYTES = 300 * 1024 * 1024
VIDEO_HOSTS = (
    "bilibili.com", "b23.tv", "douyin.com", "iesdouyin.com",
    "kuaishou.com", "ixigua.com", "xiaohongshu.com", "xhslink.com",
)
URL_RE = re.compile(r"https?://[^\s\u4e00-\u9fff]+")


def _pick_video_url(text: str) -> str:
    """从分享文案里挑出第一个视频站链接（中文和标点不会粘上去）。"""
    for raw in URL_RE.findall(text or ""):
        url = raw.rstrip("，。！？、；：)）】」』…\"'")
        if any(h in url for h in VIDEO_HOSTS):
            return url
    return ""


def _dbg(msg: str) -> None:
    try:
        with DEBUG_LOG.open("a", encoding="utf-8") as f:
            f.write(f"[{_dt.datetime.now():%m-%d %H:%M:%S}] {msg}\n")
    except Exception:  # noqa: BLE001
        pass


_WHISPER = None
_WHISPER_LOCK = threading.Lock()
WHISPER_ROOT = DATA_DIR / "models"


def _get_whisper(model_name: str = "medium"):
    r"""懒加载 whisper，加载一次后常驻（medium 约 1.5GB，首次加载 10~20 秒）。

    优先用数据目录下 faster-whisper-<名称> 里的本地模型（从 ModelScope 下的，
    镜像速度快）；没有才退回按名称去 HuggingFace 拉。
    """
    global _WHISPER
    with _WHISPER_LOCK:
        if _WHISPER is None:
            from faster_whisper import WhisperModel

            local = WHISPER_ROOT / f"faster-whisper-{model_name}"
            target = str(local) if (local / "model.bin").exists() else model_name
            _dbg(f"加载 whisper({target}) …")
            _WHISPER = WhisperModel(target, device="cpu", compute_type="int8")
            _dbg("whisper 就绪")
        return _WHISPER


def _transcribe(media: Path, model_name: str = "medium") -> str:
    """把音轨转成文字。media 可以是视频，也可以是单独下的音频文件。

    两个坑记一下：
    - vad_filter 在这个环境里会把整段判成静音（onnxruntime 版本问题），所以关掉；
    - beam_size=1 比默认快一倍，效果差别不大。
    """
    try:
        model = _get_whisper(model_name)
        segments, _info = model.transcribe(
            str(media),
            language=None,               # 自动检测，中英混合时不会被硬按成中文
            vad_filter=False,            # 见上面注释
            condition_on_previous_text=False,
            beam_size=1,
        )
        return "".join(seg.text for seg in segments).strip()
    except Exception as exc:  # noqa: BLE001
        _dbg(f"转写失败: {exc}")
        return ""


def _extract_frames(video: Path, count: int = 8):
    """按画面变化采样：先密集取候选帧，再挑变化最大的那些，按时间排序。"""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError("视频打不开（格式不支持或文件损坏）")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    duration = total / fps if fps > 0 else 0
    # 约 4 秒一帧，下限 6、上限 24（再多会糊）
    if duration > 0:
        count = min(24, max(6, int(duration / 4) + 1))

    if total <= 0:
        # 读不到帧数，顺序读
        frames, times = [], []
        step = max(1, int(fps * 3))
        idx = 0
        while len(frames) < count:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                frames.append(frame)
                times.append(idx / fps)
            idx += 1
        cap.release()
        return frames, duration, times

    # 候选点：目标帧数的 4 倍，均匀铺开
    cand_n = max(count * 4, 16)
    candidates = []
    prev_gray = None
    for i in range(cand_n):
        pos = int(total * i / cand_n)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos - 1))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        diff = 999.0 if prev_gray is None else float(np.mean(cv2.absdiff(gray, prev_gray)))
        candidates.append((diff, pos, frame))
        prev_gray = gray

    if not candidates:
        cap.release()
        return [], duration, []

    # 变化最大的 count 个，按时间排回去
    picked = sorted(candidates, key=lambda x: -x[0])[:count]
    picked.sort(key=lambda x: x[1])
    frames = [p[2] for p in picked]
    times = [p[1] / fps for p in picked]
    cap.release()
    return frames, duration, times


def _build_grid(frames, times=None, cols: int = 2, tile_w: int = 640):
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    th = max(1, int(h * tile_w / w))
    tiles = [cv2.resize(f, (tile_w, th), interpolation=cv2.INTER_AREA) for f in frames]
    rows = (len(tiles) + cols - 1) // cols
    canvas = np.zeros((rows * th, cols * tile_w, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas[r * th:(r + 1) * th, c * tile_w:(c + 1) * tile_w] = tile
        label = f"#{i + 1}"
        if times and i < len(times):
            t = int(times[i])
            label += f" {t // 60}:{t % 60:02d}"
        cv2.rectangle(canvas, (c * tile_w + 6, r * th + 6),
                      (c * tile_w + 150, r * th + 38), (0, 0, 0), -1)
        cv2.putText(canvas, label, (c * tile_w + 12, r * th + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    return canvas


@register(PLUGIN_NAME, "视频之眼", "抽帧交给视觉模型看视频", "1.1.0")
class VideoEyesPlugin(Star):
    def __init__(self, context: Context, config: dict | None = None) -> None:
        super().__init__(context)
        self.config = config or {}
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        self._seen: set[str] = set()
        self._clean_old()

    def _clean_old(self, hours: int = 24) -> None:
        cutoff = time.time() - hours * 3600
        for p in TEMP_ROOT.glob("*"):
            try:
                if p.stat().st_mtime < cutoff:
                    p.unlink()
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _read_growth_self() -> str:
        p = Path(os.environ.get("VIDEO_EYES_SELF_MD", str(DATA_DIR / "self.md")))
        try:
            return p.read_text(encoding="utf-8").strip() if p.exists() else ""
        except Exception:  # noqa: BLE001
            return ""

    # ---------- 下载 ----------
    # 约定：返回 (视频文件, 音频文件或 None)。B站这类分离流会给单独的音频；QQ 视频自带音轨则为 None。
    async def _fetch_video(self, url: str) -> tuple[Path | None, Path | None]:
        direct_suffix = (".mp4", ".webm", ".mkv", ".flv", ".mov", ".m4v")
        if url.lower().split("?")[0].endswith(direct_suffix):
            return await self._download(url), None
        return await self._download_ytdlp(url)

    async def _download(self, url: str) -> Path | None:
        name = hashlib.md5(url.encode()).hexdigest()[:12]
        out = TEMP_ROOT / f"dl_{name}.mp4"
        if out.exists() and out.stat().st_size > 1024:
            return out
        _dbg(f"直链下载: {url[:120]}")
        try:
            async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
                async with client.stream("GET", url, headers={"User-Agent": "Mozilla/5.0"}) as resp:
                    resp.raise_for_status()
                    size = 0
                    with out.open("wb") as f:
                        async for chunk in resp.aiter_bytes(65536):
                            size += len(chunk)
                            if size > MAX_BYTES:
                                raise RuntimeError("视频太大（超过 300MB）")
                            f.write(chunk)
            _dbg(f"直链下载完成: {out.stat().st_size} bytes")
            return out
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[video_eyes] 下载失败: {exc}")
            _dbg(f"直链下载失败: {exc}")
            try:
                out.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass
            return None

    async def _download_ytdlp(self, url: str) -> tuple[Path | None, Path | None]:
        """B站/抖音：视频流和音频流是分开的。

        没装 ffmpeg 没法合并，所以分两次下：视频给抽帧，音频给 whisper。
        """
        name = hashlib.md5(url.encode()).hexdigest()[:12]
        out = TEMP_ROOT / f"yt_{name}.mp4"
        audio = TEMP_ROOT / f"yt_{name}.m4a"
        exe = str(self.config.get("ytdlp_path") or "") or shutil.which("yt-dlp") or ""
        _dbg(f"yt-dlp 下载: {url[:120]}")
        if not exe or not Path(exe).exists():
            logger.warning("[video_eyes] 找不到 yt-dlp（装一下 pip install yt-dlp）")
            return None, None

        def _run(args):
            import subprocess

            return subprocess.run(args, capture_output=True, timeout=300)

        try:
            if not (out.exists() and out.stat().st_size > 1024):
                proc = await asyncio.to_thread(_run, [
                    exe, "--no-playlist", "--no-warnings",
                    "-f", "bv*[height<=720]/b[height<=720]/b",
                    "-o", str(out), url,
                ])
                if proc.returncode != 0 or not out.exists() or out.stat().st_size < 1024:
                    err = (proc.stderr or b"").decode("utf-8", "ignore")[-400:]
                    logger.warning(f"[video_eyes] yt-dlp 视频失败: {err}")
                    _dbg(f"yt-dlp 视频失败: {err}")
                    return None, None
                _dbg(f"yt-dlp 视频成功: {out.stat().st_size} bytes")

            # 音频单独下，失败不影响看画面
            if not (audio.exists() and audio.stat().st_size > 1024):
                proc2 = await asyncio.to_thread(_run, [
                    exe, "--no-playlist", "--no-warnings",
                    "-f", "ba", "-o", str(audio), url,
                ])
                if proc2.returncode == 0 and audio.exists() and audio.stat().st_size > 1024:
                    _dbg(f"yt-dlp 音频成功: {audio.stat().st_size} bytes")
                else:
                    err = (proc2.stderr or b"").decode("utf-8", "ignore")[-200:]
                    _dbg(f"yt-dlp 音频失败（不影响画面）: {err}")
                    audio = None
            return out, (audio if audio and audio.exists() else None)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[video_eyes] yt-dlp 异常: {exc}")
            _dbg(f"yt-dlp 异常: {exc}")
            return None, None

    # ---------- 处理管线 ----------
    async def _pipeline(
        self, event: AstrMessageEvent, src: Path, audio: Path | None = None
    ) -> str:
        # 听觉和视觉并行：whisper 在后台线程跑，视觉调用照常走。
        # B站视频的音轨是单独文件（audio），QQ 视频自带音轨则直接读 src。
        stt_task = None
        if self.config.get("stt_enable", True):
            stt_task = asyncio.create_task(
                asyncio.to_thread(
                    _transcribe,
                    audio or src,
                    str(self.config.get("stt_model", "medium")),
                )
            )

        try:
            frames, duration, times = await asyncio.to_thread(
                _extract_frames, src, int(self.config.get("frame_count", 8))
            )
        except Exception as exc:  # noqa: BLE001
            return f"抽帧失败：{exc}"
        if not frames:
            return "这个视频抽不出画面（可能是纯音频或编码不支持）"

        try:
            provider = self.context.get_using_provider(umo=event.unified_msg_origin)
        except Exception as exc:  # noqa: BLE001
            _dbg(f"取模型失败: {exc}")
            provider = None
        if provider is None:
            return "没有可用的视觉模型"

        # 分块看图：每块 6~8 帧拼一张，避免整张图被压缩
        chunk = 6 if len(frames) <= 16 else 8
        blocks = [
            (frames[i:i + chunk], times[i:i + chunk])
            for i in range(0, len(frames), chunk)
        ]
        _dbg(f"时长{duration:.0f}s 帧数{len(frames)} 分{len(blocks)}块")

        parts: list[str] = []
        for bi, (bf, bt) in enumerate(blocks, 1):
            if not bf:
                continue
            grid = _build_grid(bf, bt, int(self.config.get("columns", 2)))
            img_path = TEMP_ROOT / f"grid_{uuid.uuid4().hex[:8]}.jpg"
            cv2.imwrite(str(img_path), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            span = f"{bt[0]:.0f}~{bt[-1]:.0f} 秒" if bt else ""
            _dbg(
                f"第{bi}块 {img_path.name} {img_path.stat().st_size // 1024}KB "
                f"{len(bf)}帧 {grid.shape[1]}x{grid.shape[0]} {span}"
            )
            prompt = (
                f"这是一段视频的第 {bi}/{len(blocks)} 张截帧图，覆盖大约 {span}，"
                f"共 {len(bf)} 帧，每格左上角标了它在视频里的时间点。\n"
                "请回答：\n"
                "1) 画面里有谁、在做什么（多人时分别编号，同一人跨格用同一编号）\n"
                "2) 出现的字幕、界面文字、招牌，尽量原样写\n"
                "3) 这几个时间点之间发生了什么变化\n"
                "铁律：只写你真正看到的，看不清就说看不清，不要编。中文，200 字以内。"
            )
            try:
                resp = await provider.text_chat(prompt=prompt, image_urls=[str(img_path)])
                piece = (resp.completion_text or "").strip()
                if piece:
                    parts.append(f"[{span}]\n{piece}")
            except Exception as exc:  # noqa: BLE001
                _dbg(f"第{bi}块视觉调用失败: {exc}")
            finally:
                if not self.config.get("keep_frames", False):
                    try:
                        img_path.unlink(missing_ok=True)
                    except Exception:  # noqa: BLE001
                        pass

        text = "\n\n".join(parts)
        if not text:
            return "看视频失败：视觉模型没有返回内容"

        # 等听觉那边收工（它通常比视觉慢，跑在后台线程里）
        transcript = ""
        if stt_task is not None:
            try:
                transcript = await stt_task
            except Exception as exc:  # noqa: BLE001
                _dbg(f"转写任务异常: {exc}")
            _dbg(f"转写完成 {len(transcript)} 字")

        if text:
            try:
                self_md = self._read_growth_self()
                sys_prompt = (
                    "你正在跟主人私聊，用你平时的口吻说人话：短、自然，不要'以下是''综上'这种腔调，"
                    "不要复述观察报告的结构。\n"
                )
                if self_md:
                    sys_prompt += f"\n你此刻的自我认知：\n{self_md[:800]}\n"
                sound_part = (
                    "\n【转写读到的话】\n"
                    f"{transcript[:1500]}\n"
                    "（这是语音识别出的文字，不是你真听见的声音，可能有错字）"
                    if transcript
                    else "\n（这段视频没有转写出可用的话）"
                )
                resp2 = await provider.text_chat(
                    prompt=(
                        f"你刚处理完一段约 {duration:.0f} 秒的视频。两段材料的来源不同，别混：\n"
                        f"【截图看到的】\n{text}\n"
                        f"{sound_part}\n\n"
                        "用你自己的口气告诉主人这段视频讲了什么，1~3 句，口语，别照搬原文；"
                        "画面和内容对不上时以截图为准；"
                        "被问起来源时要如实说：画面是截图，声音是转写文字，你没有真实的视听。"
                    ),
                    system_prompt=sys_prompt,
                )
                t2 = (resp2.completion_text or "").strip()
                if t2:
                    text = t2
            except Exception as exc:  # noqa: BLE001
                _dbg(f"转述失败，用原文: {exc}")

        if not self.config.get("keep_frames", False):
            try:
                if src.name.startswith(("dl_", "yt_")):
                    src.unlink(missing_ok=True)
                if audio is not None:
                    audio.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass
        return text

    # ---------- 触发 ----------
    async def _component_to_file(self, comp) -> Path | None:
        """把 Video 组件落成一个本地文件：url / base64 / file:// / 本地路径都试。"""
        import base64 as _b64
        from urllib.request import url2pathname

        candidates = []
        for attr in ("url", "file", "path"):
            v = getattr(comp, attr, None)
            if isinstance(v, str) and v:
                candidates.append((attr, v))
        for attr, c in candidates:
            try:
                if c.startswith("base64://"):
                    raw = _b64.b64decode(c[len("base64://"):])
                    out = TEMP_ROOT / f"c_{uuid.uuid4().hex[:8]}.mp4"
                    out.write_bytes(raw)
                    _dbg(f"base64 视频落地({attr}): {len(raw)} bytes")
                    return out
                if c.startswith("data:"):
                    raw = _b64.b64decode(c.split(",", 1)[1])
                    out = TEMP_ROOT / f"c_{uuid.uuid4().hex[:8]}.mp4"
                    out.write_bytes(raw)
                    _dbg(f"data: 视频落地({attr}): {len(raw)} bytes")
                    return out
                if c.startswith("file://"):
                    p = Path(url2pathname(c[7:]))
                    if p.exists():
                        return p
                    continue
                if c.startswith("http"):
                    got, _audio = await self._fetch_video(c)
                    if got:
                        return got
                    continue
                p = Path(c)
                if p.exists():
                    return p
            except Exception as exc:  # noqa: BLE001
                _dbg(f"组件解析失败 {attr}={c[:60]}: {exc}")
        _dbg(f"视频组件无可下载源: {[a for a, _ in candidates]}")
        return None

    @filter.event_message_type(filter.EventMessageType.PRIVATE_MESSAGE, priority=80)
    async def auto_watch(self, event: AstrMessageEvent):
        umo = event.unified_msg_origin
        raw = (event.message_str or "").strip()
        prefix = str(self.config.get("scope_prefix", DEFAULT_SCOPE))
        if prefix and not umo.startswith(prefix):
            return
        if raw.lower().startswith(("video", "/video")):
            return

        segs = []
        video_comp = None
        for comp in event.get_messages():
            info = {"type": type(comp).__name__}
            for attr in ("file", "url", "path"):
                v = getattr(comp, attr, None)
                if v:
                    info[attr] = str(v)[:120]
            segs.append(info)
            if type(comp).__name__ == "Video" and video_comp is None:
                video_comp = comp
        _dbg(f"自动检查 raw={raw[:60]!r} 段={segs}")

        src: Path | None = None
        audio: Path | None = None
        key = ""
        if video_comp is not None:
            key = hashlib.md5(
                (getattr(video_comp, "file", "") or raw).encode()
            ).hexdigest()
            if key in self._seen:
                return
            self._seen.add(key)
            src = await self._component_to_file(video_comp)
            if src is None:
                yield event.plain_result("视频收到了，但没拿到能用的文件（细节我记日志了）")
                return
        else:
            url = _pick_video_url(raw)
            if not url:
                return
            key = hashlib.md5(url.encode()).hexdigest()
            if key in self._seen:
                return
            self._seen.add(key)
            _dbg(f"链接: {url}")
            src, audio = await self._fetch_video(url)
            if src is None:
                yield event.plain_result("这个链接下不下来，可能是需要登录或已失效")
                return

        result = await self._pipeline(event, src, audio)
        if result:
            yield event.plain_result(result)

    @filter.command("video")
    async def video_cmd(self, event: AstrMessageEvent):
        raw = (event.message_str or "").strip()
        arg = raw
        low = raw.lower()
        for token in ("/video", "video"):
            if low.startswith(token):
                arg = raw[len(token):]
                break
        arg = arg.strip().strip("/").strip().strip('"').strip("'")
        m = re.search(r"https?://[^\s]+", arg)
        if m:
            arg = m.group(0).rstrip("，。！？、)）】")
        _dbg(f"命令 /video arg={arg!r}")

        url = ""
        src: Path | None = None
        audio: Path | None = None
        if arg.startswith(("http://", "https://")):
            url = arg
        elif arg:
            p = Path(arg)
            if not p.exists():
                yield event.plain_result("路径不存在")
                return
            src = p
        else:
            for comp in event.get_messages():
                if type(comp).__name__ == "Video":
                    src = await self._component_to_file(comp)
                    break
            if src is None:
                yield event.plain_result("这条消息里没有可用的视频，直接把视频或链接发给我就行")
                return

        if src is None:
            src, audio = await self._fetch_video(url)
        if src is None or not src.exists():
            yield event.plain_result("视频下载失败")
            return
        result = await self._pipeline(event, src, audio)
        if result:
            yield event.plain_result(result)
