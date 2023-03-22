"""
Requirements:
    1) CocoDataset==0.1.2
    2) wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    3) unzip ./annotations_trainval2017.zip
"""

from coco_dataset import coco_dataset_download as cocod


def download_COCO_class_instances(
    classes: list[str],
    count: int,
    annotations_path: str,
):
    """
    Download count instances of images containing an object from the classes.

    We reduce the count by one because the count is incremented by one by this package.
    """
    for i in classes:
        # call the download function
        cocod.coco_dataset_download(i, count - 1, annotations_path)


if __name__ == "__main__":
    download_COCO_class_instances(
        ["bottle", "cup", "sports ball"],
        1,
        "./annotations/instances_train2017.json",
    )
