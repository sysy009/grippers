"""lerobot 의 배치 영상 인코딩을 동작하는 구현으로 갈아끼운다.

## 왜 필요한가

녹화 루프는 30Hz 예산이 33.3ms 다. 여기에 영상 인코딩을 얹으면 안 된다
(2026-09-02 실측, 1280x720):

    PNG 를 비동기 스레드로 T9 에 쓰기      메인 루프 비용   0.0 ms
    streaming_encoding 의 feed_frame      메인 루프 비용  21.5 ms

`feed_frame` 은 큐에 넣기만 하는데도(`video_utils.py:816`) 21.5ms 다.
인코더 스레드가 43~46Hz 밖에 못 내서 큐(maxsize 30)가 차 있고, 메인 루프가
거기서 대기하기 때문이다. 실제 녹화에서는 카메라·모터·rerun 부하까지 얹혀
**3.6Hz** 까지 떨어졌다. 그러니 PNG 로 찍고 인코딩은 나중에 해야 한다.

그런데 `--dataset.video_encoding_batch_size` 를 1보다 크게 주면, 즉 인코딩을
끝으로 미루면, lerobot 의 배치 경로가 **반드시 죽는다.**

## 원본이 깨진 곳

`_batch_save_episode_video()` 와 그것이 부르는 `_save_episode_video()` 는
녹화가 아니라 "이미 저장된 데이터셋"을 전제로 쓰여 있다.

1. `lerobot_dataset.py:1372` — `self.meta.episodes[start_episode]`
   `meta.episodes` 는 녹화 중 내내 `None` 이다. `_flush_metadata_buffer()` 가
   parquet 에는 쓰지만 그 속성은 채우지 않는다(`:118-145`).
   -> `TypeError: 'NoneType' object is not subscriptable`  **여기서 죽는다**

2. `:1375` — 아직 열려 있는 ParquetWriter 의 파일을 `pd.read_parquet` 한다.

3. `:1519-1544` — 영상을 어느 파일에 붙일지를 `meta.latest_episode` 로 정하는데,
   그건 마지막으로 **저장된** 회차(=19번)이지 마지막으로 **인코딩된** 회차가
   아니다. 배치에서는 19번이 늘 미인코딩이라 매 회차가 "새 데이터셋" 분기로
   빠진다.

죽는 자리가 `with VideoEncodingManager` 안이라 바깥 finally 의 `finalize()` 는
그대로 돈다. 그래서 info.json 도 stats.json 도 parquet 도 멀쩡한데 videos/ 만
없고, **열면 KeyError 가 나는** 데이터셋이 남는다. 녹화가 성공한 것처럼
보인다 - 2026-09-02 에 룩 20회를 그렇게 날릴 뻔했다.

## 무엇을 하나

`_batch_save_episode_video` 를 통째로 갈아끼운다. 회차마다 PNG 를 mp4 로
인코딩해 하나의 영상 파일에 이어붙이고, 타임스탬프와 인덱스를 직접 계산해
에피소드 parquet 에 쓴다. 규칙은 lerobot 과 같다 - `video_files_size_in_mb`
를 넘으면 다음 파일로 넘어간다.

같은 알고리즘을 `encode_missing_videos.py` 가 이미 룩 20회 복구에 썼고,
합성 데이터셋 20회로도 검증했다(타임스탬프 연속, 단일 mp4, 로드 정상).

⚠️ 그래도 이게 20회를 다 찍은 **뒤에** 도는 코드라는 점은 변하지 않는다.
   여기서 죽어도 PNG 는 남으므로 `encode_missing_videos.py` 로 살릴 수 있다.
"""
from __future__ import annotations

import shutil
from pathlib import Path


def patch_batch_encode() -> bool:
    """`LeRobotDataset._batch_save_episode_video` 를 교체한다."""
    import pandas as pd
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, _encode_video_worker
    from lerobot.datasets.utils import (
        DEFAULT_EPISODES_PATH,
        get_file_size_in_mb,
        update_chunk_file_indices,
    )
    from lerobot.datasets.video_utils import concatenate_video_files, get_video_duration_in_s

    if getattr(LeRobotDataset, "_batch_encode_patched", False):
        return False

    def batch_save(self, start_episode: int, end_episode: int | None = None) -> None:
        import logging

        if end_episode is None:
            end_episode = self.num_episodes

        # 원본이 첫 줄에서 죽는 이유. 버퍼를 내리고 writer 를 닫아야 에피소드
        # parquet 이 완성되고, 그래야 읽어서 영상 컬럼을 붙일 수 있다.
        self.meta._close_writer()

        # 에피소드 메타는 **여러 파일로 갈라질 수 있다.** 이어찍기(--resume)를
        # 하면 새 회차가 file-001 부터 시작한다. file-000 만 보면 그 회차들의
        # 영상 컬럼이 조용히 사라진다 - 인코딩은 됐는데 타임스탬프가 없어
        # 데이터셋을 열 수 없게 된다(2026-09-03 축구공에서 그랬다).
        # 그래서 회차마다 그 행이 들어 있는 파일을 찾아 거기에 쓴다.
        ep_files = sorted((self.root / "meta" / "episodes").rglob("*.parquet"))
        dfs = {f: pd.read_parquet(f) for f in ep_files}
        owner = {int(ep): f for f, df in dfs.items() for ep in df["episode_index"]}

        for vk in self.meta.video_keys:
            cols = [f"videos/{vk}/{c}" for c in
                    ("chunk_index", "file_index", "from_timestamp", "to_timestamp")]
            for df in dfs.values():
                for c in cols:
                    if c not in df.columns:
                        df[c] = pd.NA
            # chunk/file 은 정수, 타임스탬프는 실수여야 한다. 컬럼이 전부 NA 인
            # 채로 두면 pandas 가 실수로 추론해 인덱스가 float 이 되고
            # (`video_path.format` 이 'd' 포맷으로 죽는다), 반대로 정수로
            # 추론되면 대입이 소리 없이 잘려 20.367 이 20 이 된다.
            for df in dfs.values():
                for c in cols[2:]:
                    df[c] = df[c].astype("double[pyarrow]")

            # 이미 인코딩된 회차가 있으면(이어찍기) 그 뒤에 붙인다.
            done = pd.concat([df[df[cols[0]].notna()] for df in dfs.values()])
            if len(done):
                last = done.sort_values("episode_index").iloc[-1]
                chunk_idx, file_idx = int(last[cols[0]]), int(last[cols[1]])
                cum = float(last[cols[3]])
            else:
                chunk_idx = file_idx = 0
                cum = 0.0

            for ep in range(start_episode, end_episode):
                logging.info(f"Encoding videos for episode {ep}")
                temp = _encode_video_worker(vk, ep, self.root, self.fps,
                                            self.vcodec, self._encoder_threads)
                dur = get_video_duration_in_s(temp)
                dest = self.root / self.meta.video_path.format(
                    video_key=vk, chunk_index=chunk_idx, file_index=file_idx)

                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(temp), str(dest))
                    cum = 0.0
                elif get_file_size_in_mb(dest) + get_file_size_in_mb(temp) >= self.meta.video_files_size_in_mb:
                    chunk_idx, file_idx = update_chunk_file_indices(
                        chunk_idx, file_idx, self.meta.chunks_size)
                    dest = self.root / self.meta.video_path.format(
                        video_key=vk, chunk_index=chunk_idx, file_index=file_idx)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(temp), str(dest))
                    cum = 0.0
                else:
                    concatenate_video_files([dest, temp], dest)

                path = owner[ep]
                df = dfs[path]
                row = df["episode_index"] == ep
                if not row.any():
                    raise RuntimeError(f"에피소드 {ep} 의 메타 행을 찾지 못했습니다")
                df.loc[row, cols[0]] = chunk_idx
                df.loc[row, cols[1]] = file_idx
                df.loc[row, cols[2]] = cum
                df.loc[row, cols[3]] = cum + dur
                cum += dur
                df.to_parquet(path)

                temp = Path(temp)
                if temp.exists():
                    temp.unlink()
                if temp.parent != self.root and temp.parent.is_dir() and not any(temp.parent.iterdir()):
                    temp.parent.rmdir()

            # video.height/width 등은 0번 영상에서 읽는다.
            if start_episode == 0:
                self.meta.update_video_info(vk)

        # chunk/file 인덱스는 정수로 되돌린다. 전부 NA 였던 컬럼에 정수를 넣으면
        # pandas 가 float64 로 남겨 두는데, 그러면 데이터셋을 열 때
        # `video_path.format(...)` 이 "Unknown format code 'd' for float" 로 죽는다.
        for vk in self.meta.video_keys:
            for path, df in dfs.items():
                for c in (f"videos/{vk}/chunk_index", f"videos/{vk}/file_index"):
                    if c in df.columns and df[c].notna().all():
                        df[c] = df[c].astype("int64")
                df.to_parquet(path)

        from lerobot.datasets.utils import load_episodes, write_info
        write_info(self.meta.info, self.meta.root)
        self.meta.episodes = load_episodes(self.root)

    LeRobotDataset._batch_save_episode_video = batch_save
    LeRobotDataset._batch_encode_patched = True
    return True
