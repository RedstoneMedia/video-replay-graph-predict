from youtube_search import YoutubeSearch
import itertools
import numpy as np
from scipy.interpolate import CubicSpline, PPoly
import time
import os
import subprocess
import json
from math import floor
import random

# NOTE: This script requires yt-dlp and ffmpeg to be installed and in path

IMAGE_SIZE = 128
MIN_VIDEO_MINUTES = 1
MAX_VIDEO_MINUTES = 35
MIN_VIDEO_VIEWS = 200000

QUERY_TERMS = ["gaming", "horror", "meme", "comedy", "sketch", "minecraft", "video essay", "fortnite", "tutorial", "review", "clip"]
ADDITIONAL_TERMS = ["free", "no copyright", "meme", "compilation", "green screen", "editing", "review", "visual effect", "gameplay", "video essay", "no commentary", "tiktok"]
BAD_KEYWORDS = ["sound", "music", "lyrics", "remix", "song", "podcast"]

SAMPLES_PER_VIDEO = 30
SEQUENCE_LENGTH = 25
SEQUENCE_SAMPLE_RATE = 0.2
DOWNLOAD_RESOLUTION = "240p"
DATA_FOLDER = "C:/<PATH>/<TO>/<DATA>"


def find_good_videos(query):
    videos = YoutubeSearch(query).to_dict()
    good_videos = []
    for video in videos:
        try:
            title : str = video["title"]
            for bad_word in BAD_KEYWORDS:
                if bad_word in title.lower():
                    continue
            duration_split : list = str(video["duration"]).split(":")
            duration_seconds = 0
            for i, dur in enumerate(reversed(duration_split)):
                duration_seconds += int(dur) * 60**i
            duration_minutes = duration_seconds / 60
            views = int(video["views"].replace(".", "").replace(" Aufrufe", "").replace(" views", ""))
            if views > MIN_VIDEO_VIEWS and MIN_VIDEO_MINUTES <= duration_minutes <= MAX_VIDEO_MINUTES:
                video_id = video["id"]
                print("Found \"good\" video: ", video["title"])
                good_videos.append(f"https://www.youtube.com/watch?v={video_id}")
        except Exception as e:
            print("Error:", e)
            continue
    return good_videos


def save_good_videos():
    videos = []
    if os.path.exists("video_urls.json"):
        with open("video_urls.json", mode="r", encoding="utf-8") as f:
            videos = json.load(f)
    for term, additional in itertools.product(QUERY_TERMS, itertools.combinations(ADDITIONAL_TERMS, 2)):
        query_term = term
        b, c = additional
        if b != c:
            query_term += f" {b} {c}"
        else:
            query_term += f" {b}"
        print("Searching: ", query_term)
        videos.extend(find_good_videos(query_term))
        time.sleep(8)
        with open("video_urls.json", mode="w", encoding="utf-8") as f:
            json.dump(videos, f)


def get_interpolation_heatmap_spline(heatmap_data: list) -> PPoly:
    start_times = [d["start_time"] for d in heatmap_data]
    end_times = [d["end_time"] for d in heatmap_data]
    values = [d["value"] for d in heatmap_data]
    values = [ heatmap_data[0]["value"] ] + values + [ heatmap_data[-1]["value"] ]

    middle_times = [(pair[0] + pair[1]) / 2.0 for pair in zip(start_times, end_times)]
    times = [ heatmap_data[0]["start_time"] ] + middle_times + [ heatmap_data[-1]["end_time"] ]
    spline = CubicSpline(times, values)
    return spline


def download_with_heatmap(video_url: str):
    # Download video seq_metadata
    print("Downloading seq_metadata", video_url)
    subprocess.call(["yt-dlp", video_url, "--skip-download", "--write-info", "-o", "video"])
    output_json_path = "video.info.json"
    with open(output_json_path, mode="r", encoding="utf-8") as f:
        video_metadata : dict = json.load(f)
    # If there is no heatmap data, what's the point?
    if "heatmap" in video_metadata:
        heatmap_data = video_metadata["heatmap"]
    else:
        os.remove(output_json_path)
        print("Video had no heatmap")
        return
    # Pick good video_format_id
    video_id = video_metadata["id"]
    video_formats = video_metadata["formats"]
    video_format_id = None
    video_fps = None
    for video_format in video_formats:
        try:
            format_note = video_format["format_note"]
            audio_codec = video_format["acodec"]
            fps = video_format["fps"]
            file_extension = video_format["ext"]
        except KeyError:
            continue
        if format_note == DOWNLOAD_RESOLUTION and audio_codec == "none" and file_extension == "mp4" and fps > 29:
            video_format_id = video_format["format_id"]
            video_fps = fps
            break

    if video_format_id == None:
        os.remove(output_json_path)
        print("Could not find good video format")
        return
    # Download video and extract frames
    print("Downloading video", video_url)
    video_folder = os.path.join(DATA_FOLDER, video_id)
    if os.path.exists(video_folder):
        print("Was already downloaded", video_url)
        return
    video_output_path = f"{DATA_FOLDER}/video_{video_id}.mp4"
    subprocess.call(["yt-dlp", video_url, "-f", video_format_id, "-o", video_output_path])
    os.mkdir(video_folder)
    print("Extracting frames", video_url)
    subprocess.call(["ffmpeg", "-i", video_output_path, "-vf", f"scale={IMAGE_SIZE}:{IMAGE_SIZE}", f"{video_folder}/%04d.png"])
    # Choose some samples from the video
    heatmap_points : list = random.choices(heatmap_data, k=SAMPLES_PER_VIDEO-1)
    max_value = max(heatmap_data, key=lambda x: x["value"]) # Always add the highlight of the video
    print(f"Max value was {max_value['value']} from {max_value['start_time']}s - {max_value['end_time']}s")
    heatmap_points.append(max_value)
    heatmap_points = [dict(t) for t in {tuple(d.items()) for d in heatmap_points}]  # Dedupe, because the max_value sample might already be included

    samples = []
    all_frames = set()
    values_interpolation_spline = get_interpolation_heatmap_spline(heatmap_data)
    for heatmap_point in heatmap_points:
        start = heatmap_point["start_time"]
        end = heatmap_point["end_time"]
        duration = end -start
        value = heatmap_point["value"]
        # Pick some random frames and their interpolated values
        sequence_start = random.uniform(duration*0.2 + start, end - duration*0.2)
        frame_values = dict()
        for i in range(0, SEQUENCE_LENGTH):
            frame_time = sequence_start + i * SEQUENCE_SAMPLE_RATE
            frame_value = values_interpolation_spline(frame_time)
            frame = floor(video_fps * frame_time)
            frame_values[frame] = {
                "value": float(frame_value),
                "time": frame_time
            }
            all_frames.add(frame)
        sample = {"frame_values": frame_values, "raw": {"start": start, "end": end, "value": value}}
        samples.append(sample)
    # Save the samples
    with open(os.path.join(video_folder, "samples.json"), mode="w", encoding="utf-8") as f:
        data = {
            "samples": samples,
            "fps": video_fps,
            "duration": video_metadata["duration"]
        }
        json.dump(data, f)
    # Remove all frames, which have not been sampled
    for file in os.listdir(video_folder):
        if not file.endswith(".png"):
            continue
        frame_number = int(file.split(".png")[0])
        if not frame_number in all_frames:
            os.remove(os.path.join(video_folder, file))
    # Cleanup
    os.remove(video_output_path)
    os.remove(output_json_path)

def download_with_heatmaps():
    with open("video_urls.json", mode="r", encoding="utf-8") as f:
        videos = json.load(f)
    videos = set(videos) # Removes duplicates (which there are probably a lot of)
    print(f"Downloading {len(videos)} videos")
    for video_url in videos:
        try:
            download_with_heatmap(video_url)
        except Exception as e:
            print(f"Error while downloading video \"{video_url}\" with heatmap: {e}")

def main():
    save_good_videos()
    if input("Continue by downloading heatmaps? Y/n").strip().lower() == "Y":
        download_with_heatmaps()


if __name__ == "__main__":
    main()