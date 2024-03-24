import ast
import os
import shutil

import numpy as np
import supervisely as sly
from cv2 import connectedComponents
from dataset_tools.convert import unpack_if_archive
from supervisely.io.fs import (
    file_exists,
    get_file_name,
    get_file_name_with_ext,
    get_file_size,
)
from tqdm import tqdm

import src.settings as s


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    # Possible structure for bbox case. Feel free to modify as you needs.

    images_path = "/home/alex/DATASETS/TODO/Spacecraft/Spacecrafts/images"
    batch_size = 30

    boxes_path = "/home/alex/DATASETS/TODO/Spacecraft/Spacecrafts/all_bbox.txt"

    def get_unique_colors(img):
        unique_colors = []
        img = img.astype(np.int32)
        h, w = img.shape[:2]
        colhash = img[:, :, 0] * 256 * 256 + img[:, :, 1] * 256 + img[:, :, 2]
        unq, unq_inv, unq_cnt = np.unique(colhash, return_inverse=True, return_counts=True)
        indxs = np.split(np.argsort(unq_inv), np.cumsum(unq_cnt[:-1]))
        col2indx = {unq[i]: indxs[i][0] for i in range(len(unq))}
        for col, indx in col2indx.items():
            if col != 0:
                unique_colors.append((col // (256**2), (col // 256) % 256, col % 256))

        return unique_colors

    def create_ann(image_path):
        labels = []

        # image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = 720  # image_np.shape[0]
        img_wight = 1280  # image_np.shape[1]

        name_index = get_file_name(image_path).split("_")[-1]
        if int(name_index) > 1002:
            tag = sly.Tag(coarse_meta)
        else:
            tag = sly.Tag(fine_meta)
        bboxes_data = name_to_bbox.get(name_index)
        if bboxes_data is not None:
            for curr_data in bboxes_data:
                left = int(curr_data[2])
                top = int(curr_data[3])
                right = int(curr_data[0])
                bottom = int(curr_data[1])
                rectangle = sly.Rectangle(top=top, left=left, bottom=bottom, right=right)
                label = sly.Label(rectangle, spacecrafts)
                labels.append(label)

        mask_path = image_path.replace("images", "mask").replace(".png", "_mask.png")

        if file_exists(mask_path):
            mask_np = sly.imaging.image.read(mask_path)
            unique_colors = get_unique_colors(mask_np)
            for color in unique_colors:
                mask = np.all(mask_np == color, axis=2)
                ret, curr_mask = connectedComponents(mask.astype("uint8"), connectivity=8)
                for i in range(1, ret):
                    obj_mask = curr_mask == i
                    bitmap = sly.Bitmap(data=obj_mask)
                    if bitmap.area > 50:
                        obj_class = color_to_obj_class[color]
                        label = sly.Label(bitmap, obj_class)
                        labels.append(label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=[tag])

    body = sly.ObjClass("body", sly.Bitmap)
    solar = sly.ObjClass("solar panel", sly.Bitmap)
    antenna = sly.ObjClass("antenna", sly.Bitmap)
    spacecrafts = sly.ObjClass("spacecraft", sly.Rectangle)

    color_to_obj_class = {(255, 0, 0): solar, (0, 0, 255): antenna, (0, 255, 0): body}

    fine_meta = sly.TagMeta("fine mask", sly.TagValueType.NONE)
    coarse_meta = sly.TagMeta("coarse mask", sly.TagValueType.NONE)

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(
        obj_classes=[spacecrafts, body, solar, antenna], tag_metas=[fine_meta, coarse_meta]
    )
    api.project.update_meta(project.id, meta.to_json())

    name_to_bbox = {}
    with open(boxes_path) as f:
        content = f.read()
        name_to_bbox = ast.literal_eval(content)

    for ds_name in os.listdir(images_path):

        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        curr_images_path = os.path.join(images_path, ds_name)

        images_names = os.listdir(curr_images_path)

        progress = sly.Progress("Add {} in dataset".format(ds_name), len(images_names))

        for img_names_batch in sly.batched(images_names, batch_size=batch_size):
            images_pathes_batch = [
                os.path.join(curr_images_path, image_name) for image_name in img_names_batch
            ]
            img_infos = api.image.upload_paths(dataset.id, img_names_batch, images_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            anns_batch = [create_ann(image_path) for image_path in images_pathes_batch]
            api.annotation.upload_anns(img_ids, anns_batch)

            progress.iters_done_report(len(img_names_batch))

    return project
