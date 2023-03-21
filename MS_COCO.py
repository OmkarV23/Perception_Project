import fiftyone as fo
import fiftyone.zoo as foz

# Define the desired object classes
classes = ["bottle", "cup", "sports ball"]

# Define a filter to only include samples that contain at least one object
# with a label in the desired classes
filter = {"$or": [{"ground_truth.detections.label": c} for c in classes]}

# Define the `info` parameter to include the `bounding_box` field
info = {"bounding_box": True}

# Load the COCO 2017 training split and apply the filter and info parameters
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["detections"],
    filter=filter,
    info=info
)

# Export the dataset to COCO format
dataset.export(
    export_dir="./MS_COCO",
    dataset_type=fo.types.COCODetectionDataset,
    label_field="ground_truth",
)
