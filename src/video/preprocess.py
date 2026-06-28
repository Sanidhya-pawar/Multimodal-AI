import os
import cv2

# ==========================================
# DATASET PATH
# ==========================================

DATASET_PATH = "datasets/video/Real Life Violence Dataset"

# ==========================================
# CLASSES
# ==========================================

classes = []

video_count = 0

print("Checking Video Dataset...\n")

for class_name in sorted(os.listdir(DATASET_PATH)):

    class_path = os.path.join(
        DATASET_PATH,
        class_name
    )

    if not os.path.isdir(class_path):
        continue

    classes.append(class_name)

    videos = [
        file
        for file in os.listdir(class_path)
        if file.endswith(
            (
                ".avi",
                ".mp4"
            )
        )
    ]

    print(
        f"{class_name}: {len(videos)} videos"
    )

    video_count += len(videos)

print("\nClasses Found:")
print(classes)

print("\nTotal Classes:")
print(len(classes))

print("\nTotal Videos:")
print(video_count)

# ==========================================
# SAMPLE VIDEO INFO
# ==========================================

sample_video = None

for class_name in classes:

    class_path = os.path.join(
        DATASET_PATH,
        class_name
    )

    for file in os.listdir(class_path):

        if file.endswith(
            (
                ".avi",
                ".mp4"
            )
        ):

            sample_video = os.path.join(
                class_path,
                file
            )

            break

    if sample_video:
        break

cap = cv2.VideoCapture(sample_video)

fps = cap.get(
    cv2.CAP_PROP_FPS
)

frame_count = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)

width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

duration = frame_count / fps

print("\nSample Video Information")

print(f"Resolution : {width} x {height}")

print(f"FPS        : {fps:.2f}")

print(f"Frames     : {frame_count}")

print(f"Duration   : {duration:.2f} sec")

cap.release()

print("\nDataset Ready")