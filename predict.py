import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.interpolate import splev, splrep
from tensorflow.keras.layers import GRU, Input
from tensorflow.keras.utils import Sequence, load_img, img_to_array
from tensorflow.keras.models import Model

import json
import re
import argparse
import os
import subprocess

TEMP_FRAMES_PATH = "temp_frames"
OUT_SMOOTHING = 0.005

def norm_duration(durations: np.ndarray, avg_duration: float) -> np.ndarray:
    return durations / avg_duration

def get_video_data(path: str) -> (float, float):
    probe_output_bytes = subprocess.check_output(["ffprobe", "-i", path], stderr=subprocess.STDOUT)
    probe_output = probe_output_bytes.decode(encoding="utf-8")
    fps = float(re.findall(r"(\d+\.*\d*) fps", probe_output)[0])
    duration_string = re.findall(r"Duration: ([0-9:.]+),", probe_output)[0]
    duration = float("0." + duration_string.split(".")[1])
    for i, duration_split in enumerate(reversed(duration_string.split(":"))):
        duration += int(duration_split.split(".")[0]) * 60**i
    return fps, duration


def split_video(path: str, image_size: (int, int), every_n_secs, start=None, end=None):
    fps, duration = get_video_data(path)
    subprocess.call(["ffmpeg", "-i", path] + (["-ss", start] if start else []) + (["-to", end] if end else []) + ["-vf",f"scale={image_size[0]}:{image_size[1]}'", "-vsync", "0", f"{TEMP_FRAMES_PATH}/%04d.png"])
    # Delete frames that are not every n secs
    t = 1/fps
    frames = os.listdir(TEMP_FRAMES_PATH)
    for frame in frames:
        t += 1/fps
        if t >= every_n_secs:
            t -= every_n_secs
        else:
            frame_path = os.path.join(TEMP_FRAMES_PATH, frame)
            os.remove(frame_path)

class CustomDirectoryImageGenerator(Sequence):

    def __init__(self, folder: str, seq_metadata: np.ndarray, padding_frames, stateful=True, stateless_sequence_size=None):
        self.image_paths = os.listdir(folder)
        self.seq_metadata = seq_metadata
        self.stateful = stateful
        self.stateless_sequence_size = stateless_sequence_size
        self.image_paths.sort(key=lambda name: int(name.split(".")[0]))
        # Add some padding images at the start
        for _ in range(padding_frames):
            self.image_paths.insert(0, self.image_paths[0])

    def get_img_array(self, index: int) -> np.ndarray:
        frame_path = os.path.join(TEMP_FRAMES_PATH, self.image_paths[index])
        img = load_img(frame_path)
        img_array = img_to_array(img) / 255.0
        return img_array


    # Loads all images in batch dynamically from disk, to avoid taking up too much RAM.
    def __getitem__(self, index):
        if self.stateful:
            img_array = self.get_img_array(index)
            return ([np.array([[img_array]]), np.array([self.seq_metadata])],)
        else:
            # Stateless is shit and takes way to long. It is still there because it is closer to how the model was trained (Does make a difference in prediction output, but the difference is not that bad)
            if index % 100 == 0:
                print(f"{index}/{len(self.image_paths)}")
            seq = []
            for i in range(index, min(index + self.stateless_sequence_size, len(self.image_paths))):
                seq.append(self.get_img_array(i))
            metadata_seq = np.repeat(self.seq_metadata, len(seq))
            return ([np.array([seq]), np.array([metadata_seq])],)


    def __len__(self):
        return len(self.image_paths)


def predict_video_heatmap(model: Model, path: str, model_metadata: dict, start=None, end=None, skip_split_frames=False, no_cleanup=False, stateful=True):
    train_sequence_size = model_metadata["train_sequence_size"]
    image_size = model_metadata["image_size"]
    dataset_metadata = model_metadata["dataset_meta"]
    sequence_sample_rate = dataset_metadata["sequence_sample_rate"]
    dataset_avg_duration = dataset_metadata["avg_duration"]

    if not skip_split_frames:
        for file in os.listdir(TEMP_FRAMES_PATH):
            os.remove(os.path.join(TEMP_FRAMES_PATH, file))
        split_video(path, image_size, sequence_sample_rate, start, end)

    fps, duration = get_video_data(path)
    seq_metadata = norm_duration(np.array([duration]), dataset_avg_duration)
    print("Normalized duration: ", seq_metadata[0])

    generator = CustomDirectoryImageGenerator(TEMP_FRAMES_PATH, seq_metadata, padding_frames=train_sequence_size, stateful=stateful, stateless_sequence_size=train_sequence_size)
    n_samples = len(generator.image_paths)
    y = model.predict(generator, steps=n_samples)
    if stateful:
        y = y[train_sequence_size:] # Remove padding

    x = np.empty(y.shape[0])
    for i, img_name in enumerate(generator.image_paths[train_sequence_size:]):
        frame = int(img_name.split(".")[0])
        x[i] = frame
    x /= float(fps)

    # Cleanup
    if no_cleanup:
        return x, y
    for file in os.listdir(TEMP_FRAMES_PATH):
        os.remove(os.path.join(TEMP_FRAMES_PATH, file))
    return x, y


def get_highlights(x: np.ndarray, y: np.ndarray, median_mul = 1.2) -> list:
    threshold = float(np.median(y)) * median_mul
    highlights = []
    current_highlight_start = None
    current_highlight_value = 0.0
    for i, y in enumerate(y):
        current_time = x[i]
        in_highlight = current_highlight_start != None
        is_above_threshold = float(y) > threshold
        if not in_highlight and is_above_threshold:
            current_highlight_start = current_time

        if not in_highlight:
            continue

        if is_above_threshold:
            current_highlight_value += float(y)
        else:
            if current_highlight_value > 0.0:
                highlights.append((current_highlight_start, current_time, current_highlight_value))
            current_highlight_start = None
            current_highlight_value = 0.0

    if current_highlight_start != None:
        if current_highlight_value > 0.0:
            highlights.append((current_highlight_start, x[-1], current_highlight_value))

    highlights.sort(key=lambda x:x[2], reverse=True)
    return highlights


def graph_heatmap(x: np.ndarray, y: np.ndarray):
    smoothing = 0.4

    median_mul = 1.1
    threshold = np.median(y) * median_mul
    highlights = get_highlights(x, y, median_mul=median_mul)[:3]

    while True:
        tck = splrep(x, y, k=3, s=smoothing)
        y_smooth = splev(x, tck, der=0)

        plt.plot(x, y, "r", x, y_smooth, "b", x, np.repeat(threshold, x.shape[0]), "--y")
        for highlight in highlights:
            plt.axvline(highlight[0], color="purple")
            plt.axvline(highlight[1], color="purple")
        plt.show()
        s = input("Adjust smoothing: ")
        if len(s) == 0:
            break
        smoothing = float(s)

def to_stateful_model(model: Model, model_metadata: dict):
    width = model_metadata["image_size"][0]
    height = model_metadata["image_size"][1]
    sequential_input = Input(batch_shape=(1, 1, width, height, 3))
    metadata_input = Input(batch_shape=(1, 1, 1))
    inputs = [sequential_input, metadata_input]

    x = sequential_input
    for i in range(1, len(model.layers)):
        layer = model.layers[i]
        if layer.name == "gru":
            x = GRU(layer.units, stateful=True)(x)
        elif layer.name == "concatenate":
            x = layer([x, metadata_input])
        elif "input" in layer.name:
            pass
        else:
            x = layer(x)

    new_model = Model(inputs=inputs, outputs=x)
    new_model.set_weights(model.get_weights())
    return new_model


def load_model_metadata(model_path: str) -> dict:
    model_metadata_path = model_path.replace("_best", "").replace("_end", "").replace(".model", "")
    model_metadata_path += ".meta.json"
    with open(model_metadata_path, mode="r", encoding="utf-8") as f:
        return json.load(f)


def main():
    arg_parser = argparse.ArgumentParser("predict")
    arg_parser.add_argument("video", help="The path to the video which replay graph should be predicted", type=str)
    arg_parser.add_argument("--model", default="models/v2_smol_less_gru_med_dense_duration_more_before_best.model", help="The model that is used for prediction", type=str)
    arg_parser.add_argument("--graph", default=False, action="store_true", help="Show the replay graph")
    arg_parser.add_argument("--output", help="Where the predicted output will be saved to", type=str)
    arg_parser.add_argument("--skip_split", default=False, action="store_true", help="Use the previously split video frames")
    arg_parser.add_argument("--no_cleanup", default=False, action="store_true", help="Do not remove the video frames when done")
    arg_parser.add_argument("--stateless", default=False, action="store_true", help="Use stateless prediction. Usually slower and produces slightly worse results")
    args = arg_parser.parse_args()

    if not os.path.isdir(TEMP_FRAMES_PATH):
        os.mkdir(TEMP_FRAMES_PATH)

    model = tf.keras.models.load_model(args.model)
    model_metadata = load_model_metadata(args.model)
    stateful = not args.stateless
    if stateful:
        model = to_stateful_model(model, model_metadata)
    x, y = predict_video_heatmap(model, args.video, model_metadata, stateful=stateful, skip_split_frames=args.skip_split, no_cleanup=args.no_cleanup)

    if args.output != None:
        tck = splrep(x, y, k=3, s=OUT_SMOOTHING)
        y_smooth : np.ndarray = splev(x, tck, der=0)
        highlights = get_highlights(x, y_smooth)
        highlights_json_ready = [{"start": highlight[0], "end": highlight[1], "value": highlight[2]} for highlight in highlights]
        data = {"x": x.tolist(), "y": y_smooth.tolist(), "highlights": highlights_json_ready}
        with open(args.output, mode="w", encoding="utf-8") as f:
            json.dump(data, f)

    if args.graph:
        graph_heatmap(x, y)


if __name__ == "__main__":
    main()