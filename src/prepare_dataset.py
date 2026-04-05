import os
import shutil

# Input and output paths
input_base = "DATASET"
output_base = "dataset"

# Label mapping
label_map = {
    "1": "surprise",
    "2": "fear",
    "3": "disgust",
    "4": "happiness",
    "5": "sadness",
    "6": "anger",
    "7": "neutral"
}

# Create output folders
for split in ["train", "test"]:
    for emotion in label_map.values():
        os.makedirs(os.path.join(output_base, split, emotion), exist_ok=True)

# Copy images
for split in ["train", "test"]:
    for label, emotion in label_map.items():
        src_folder = os.path.join(input_base, split, label)
        dst_folder = os.path.join(output_base, split, emotion)

        if os.path.exists(src_folder):
            for file in os.listdir(src_folder):
                src_file = os.path.join(src_folder, file)
                dst_file = os.path.join(dst_folder, file)

                shutil.copy(src_file, dst_file)

print("✅ Dataset converted successfully!")