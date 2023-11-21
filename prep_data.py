from collect_data import DATA_FOLDER, SEQUENCE_SAMPLE_RATE
import numpy as np
import pickle
import random
import json
import os

from typing import Callable, Union

SEQUENCE_SIZE = 15
VALIDATION_SPLIT = 0.2

def save_pairs(pairs: list, path: str, metadata_post_process: Union[Callable[[np.ndarray], np.ndarray], None] = None):
    x_image_paths = [pair[0][0] for pair in pairs]
    x_metadata = np.array([pair[0][1] for pair in pairs])
    if metadata_post_process != None:
        x_metadata = metadata_post_process(x_metadata)
    y = np.array([pair[1] for pair in pairs])
    data = {
        "x_image_paths": x_image_paths,
        "x_metadata": x_metadata,
        "y": y
    }
    with open(path, mode="wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def save_dateset_metadata(avg_duration: float, max_duration: float, num_videos: int, path: str):
    dataset_metadata = {
        "avg_duration": avg_duration,
        "max_duration": float(max_duration),
        "num_videos": num_videos,
        "sequence_sample_rate": SEQUENCE_SAMPLE_RATE
    }
    with open(path, mode="w", encoding="utf-8") as f:
        json.dump(dataset_metadata, f)

def norm_duration(durations: np.ndarray, avg_duration: float) -> np.ndarray:
    return durations / avg_duration

def main():
    input_output_pairs = []
    durations = []
    discarded_sequences = 0
    for sample_folder_name in os.listdir(DATA_FOLDER):
        sample_folder_path = os.path.join(DATA_FOLDER, sample_folder_name)
        print("Processing", sample_folder_path)
        samples_json_file_path = os.path.join(sample_folder_path, "samples.json")
        if not os.path.exists(samples_json_file_path):
            continue
        with open(samples_json_file_path, mode="r", encoding="utf-8") as f:
            data = json.load(f)
        samples = data["samples"]
        video_duration = data["duration"]
        for sequence in samples:
            frame_values: dict = sequence["frame_values"]
            sequence_x = []
            sequence_y = []
            for frame, frame_data in sorted(frame_values.items()):
                frame_path = os.path.join(sample_folder_path, f"{frame.zfill(4)}.png")
                if not os.path.exists(frame_path):
                    continue
                value = frame_data["value"]
                rel_frame_path = os.path.join(sample_folder_name, f"{frame.zfill(4)}.png")
                sequence_x.append(rel_frame_path)
                sequence_y.append(value)
                if len(sequence_y) >= SEQUENCE_SIZE:
                    break
            else:
                print(f"Warning: Sample did not have enough items in the sequence. Should be {SEQUENCE_SIZE}, but was {len(sequence_y)}")
                discarded_sequences += 1
                continue
            duration_data = np.repeat(float(video_duration), SEQUENCE_SIZE) # float(video_duration) if inserting before GRU
            input_output_pairs.append(((sequence_x, duration_data), sequence_y))
        durations.append(video_duration)

    print("Discarded sequences:", discarded_sequences)
    print("Number of videos:", len(durations))
    avg_duration = np.average(np.array(durations))
    max_duration = np.max(np.array(durations))
    print(f"Average duration: {round(avg_duration, ndigits=2)}s")
    print(f"Maximum duration: {round(max_duration, ndigits=2)}s")

    random.seed(1337)
    random.shuffle(input_output_pairs)
    validation_size = round(len(input_output_pairs) * VALIDATION_SPLIT)
    validation = input_output_pairs[:validation_size]
    train = input_output_pairs[validation_size:]
    print(f"Train size: {len(train)}, Validation size: {len(validation)}")

    save_dateset_metadata(avg_duration, max_duration, len(durations), "dataset_meta.json")
    save_pairs(validation, "validation.pickle", metadata_post_process=lambda x: norm_duration(x, max_duration))
    save_pairs(train, "train.pickle", metadata_post_process=lambda x: norm_duration(x, max_duration))


if __name__ == "__main__":
    main()