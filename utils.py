import imageio
import os
import typing as T
import numpy as np

# @title Utility code for displaying videos
def write_video(
    filepath: os.PathLike,
    frames: T.Iterable[np.ndarray],
    fps: int = 60,
    macro_block_size: T.Optional[int] = None,
    quality: int = 10,
    verbose: bool = False,
    **kwargs,
):
    """
    Saves a sequence of frames as a video file.

    Parameters:
    - filepath (os.PathLike): Path to save the video file.
    - frames (Iterable[np.ndarray]): An iterable of frames, where each frame is a numpy array.
    - fps (int, optional): Frames per second, defaults to 60.
    - macro_block_size (Optional[int], optional): Macro block size for video encoding, can affect compression efficiency.
    - quality (int, optional): Quality of the output video, higher values indicate better quality.
    - verbose (bool, optional): If True, prints the file path where the video is saved.
    - **kwargs: Additional keyword arguments passed to the imageio.get_writer function.

    Returns:
    None. The video is written to the specified filepath.
    """

    with imageio.get_writer(
        filepath, fps=fps, macro_block_size=macro_block_size, quality=quality, **kwargs
    ) as video:
        if verbose:
            print("Saving video to:", filepath)
        for frame in frames:
            video.append_data(frame)