import os

DATASET_PATH = "datasets/audio/speech_commands"

print("Checking dataset...\n")

classes = []

for item in os.listdir(DATASET_PATH):

    full_path = os.path.join(
        DATASET_PATH,
        item
    )

    if os.path.isdir(full_path):

        classes.append(item)

print("Classes Found:")
print(classes)

print("\nTotal Classes:")
print(len(classes))

print("\nSample Counts:\n")

for cls in classes:

    cls_path = os.path.join(
        DATASET_PATH,
        cls
    )

    count = len([
        f for f in os.listdir(cls_path)
        if f.endswith(".wav")
    ])

    print(f"{cls}: {count}")