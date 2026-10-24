# 视频之眼 · AstrBot 插件

让 Bot 真的"看懂"视频：**画面靠抽帧，声音靠本地语音识别**，最后用它自己的口吻讲给你。

> 不是"把视频发出来"，是"看完之后告诉你里面发生了什么"。

## 它能做什么

- 你发一段视频 → 它描述画面里有什么、谁在做什么、字幕写了什么
- 你发 B站 / 抖音链接 → 自动下载、抽帧、转写（`yt-dlp`）
- 视频里有说话 → 本地 `faster-whisper` 转成文字，和画面一起理解
- 结果会标注来源：**哪些是截图看到的，哪些是转写读到的**，不让模型把两者混为一谈

## 怎么做的

```
视频文件
  ├─ 抽帧：约 4 秒一帧（6~24 帧），按画面变化挑选关键帧
  ├─ 看图：每 6~8 帧拼一张图，分块送视觉模型，避免整张图被压缩糊掉
  └─ 听声：faster-whisper 转写（自动识别语言），只下音频流时也能单独处理
        ↓
  两路结果标注来源 → 交回主模型，用当前人格口吻转述
```

## 安装

1. 把本目录放到 `AstrBot/data/plugins/astrbot_plugin_video_eyes/`
2. 装依赖：

```bash
pip install -r requirements.txt
```

3. 重启 AstrBot，面板里会出现「视频之眼」

### 关于语音模型

`faster-whisper` 首次使用会去 HuggingFace 下载模型，国内基本拉不动。建议：

- 从 [ModelScope](https://modelscope.cn/models/pengzhendong/faster-whisper-medium) 下载
  `model.bin`、`config.json`、`tokenizer.json`、`vocabulary.txt`
- 放到插件的 `data/models/faster-whisper-medium/` 目录下
- 插件会优先用本地模型，找不到才去下载

## 配置

| 配置项 | 默认 | 说明 |
| --- | --- | --- |
| `frame_count` | 8 | 抽帧基准值（实际按 4 秒一帧自动算） |
| `columns` | 2 | 拼图列数 |
| `stt_enable` | true | 是否开启语音转写 |
| `stt_model` | medium | tiny / base / small / medium |
| `keep_frames` | false | 是否保留抽帧图片 |
| `scope_prefix` | 空 | 只在这个 UMO 前缀的会话生效，空=所有私聊 |
| `ytdlp_path` | 空 | yt-dlp 可执行文件路径，空则从 PATH 找 |

## 我踩过的坑（照抄能省一晚上）

1. **`vad_filter=True` 会把整段音频判成静音**，转写结果永远是空字符串。这个环境下（onnxruntime 版本问题）必须关掉。
2. **B站的音频和视频是两条流**，`yt-dlp -f bv*` 只下画面，没有声音。没装 ffmpeg 就没法合并，所以要分两次下：视频给抽帧、音频给 whisper。
3. **`av` 要锁 13.x**。14+ 的 `av.open()` 没有 `metadata_errors` 参数，faster-whisper 会直接抛 `TypeError`。
4. **ModelScope 下模型比 HuggingFace 镜像快 5 倍**（实测 10.9 MB/s vs 2.3 MB/s），而且不会遇到 `xet` 传输 401。
5. **QQ 视频消息有时只有 `file` 没有 `url`**。只认 `url` 会拿到空壳；正确做法是依次尝试 `url` → `file` → `path`，并支持 `base64://`、`data:`、`file://`、本地路径。
6. **一张大拼图不如分块看**：模型看图的像素有预算，16 帧塞一张 2560×1440 会被压缩到每帧约 380px，糊得没法认字。分成 2~3 张就清楚了。

## 限制

- 抽帧是"每隔几秒看一眼"，不是连续看视频。连贯动作（先抬手再转身）它看不出来。
- 不支持 silk 语音输入（QQ 语音消息的格式），只处理视频音轨。
- 需要视觉模型：你的对话 provider 必须支持 `image` 输入。

## License

MIT
