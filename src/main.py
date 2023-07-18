import os
from pathlib import Path
from time import sleep
from typing import Tuple

import requests
import supervisely as sly
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv(os.path.expanduser("~/supervisely.env"))
load_dotenv("local.env")

# get env variables
GLOBAL_TIMEOUT = 1  # seconds
AGENT_ID = 230  # agent id to run training on
PROJECT_ID = sly.env.project_id()
DATASET_ID = sly.env.dataset_id()
TEAM_ID = sly.env.team_id()
WORKSPACE_ID = sly.env.workspace_id()
TASK_TYPE = "object detection"  # you can choose "instance segmentation" or "pose estimation"
DATA_DIR = sly.app.get_data_dir()
image_path = "/Users/almaz/projects/data/test/image.jpeg"


def train_model(api: sly.Api) -> Tuple[str, str]:
    train_app_name = "supervisely-ecosystem/yolov8/train"

    module_id = api.app.get_ecosystem_module_id(train_app_name)
    module_info = api.app.get_ecosystem_module_info(module_id)
    project_name = api.project.get_info_by_id(PROJECT_ID).name

    sly.logger.info(f"Starting AutoTrain for application {module_info.name}")

    params = module_info.get_arguments(images_project=PROJECT_ID)

    session = api.app.start(
        agent_id=AGENT_ID,
        module_id=module_id,
        workspace_id=WORKSPACE_ID,
        description=f"AutoTrain session for {module_info.name}",
        task_name="AutoTrain/train",
        params=params,
    )

    task_id = session.task_id
    domain = sly.env.server_address()
    token = api.task.get_info_by_id(task_id)["meta"]["sessionToken"]
    post_shutdown = f"{domain}/net/{token}/sly/shutdown"

    while not api.task.get_status(task_id) is api.task.Status.STARTED:
        sleep(GLOBAL_TIMEOUT)
    else:
        sleep(10)  # still need a time after status changed

    sly.logger.info(f"Session started: #{task_id}")

    # 📗 You can set any parameters you want to customize training in the data field
    api.task.send_request(
        task_id,
        "auto_train",
        data={
            "project_id": PROJECT_ID,
            # "dataset_ids": [DATASET_ID], # optional (specify if you want to train on specific datasets)
            "task_type": TASK_TYPE,
            "train_mode": "finetune", # finetune / scratch
            "n_epochs": 100,
            "patience": 50,
            "batch_size": 16,
            "input_image_size": 640,
            "optimizer": "AdamW", # AdamW, Adam, SGD, RMSProp
            "n_workers": 8,
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "amp": "true",
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
        }, # 📗 train paramaters
        timeout=10e6,
    )

    team_files_folder = Path("/yolov8_train") / TASK_TYPE / project_name / str(task_id)
    weights = Path(team_files_folder) / "weights"
    best = None

    while best is None:
        sleep(GLOBAL_TIMEOUT)
        if api.file.dir_exists(TEAM_ID, str(weights)):
            for filename in api.file.listdir(TEAM_ID, str(weights)):
                if os.path.basename(filename).startswith("best"):
                    best = str(weights / filename)
                    sly.logger.info(f"Checkpoint founded : {best}")

    requests.post(post_shutdown)

    return str(team_files_folder), best


def download_weight(api: sly.Api, remote_weight_path):
    # download best weight from Supervisely Team Files
    weight_name = os.path.basename(remote_weight_path)
    weight_dir = os.path.join(DATA_DIR, "weights")
    local_weight_path = os.path.join(weight_dir, weight_name)
    if sly.fs.dir_exists(weight_dir):
        sly.fs.remove_dir(weight_dir)

    api.file.download(TEAM_ID, remote_weight_path, local_weight_path)
    return local_weight_path


def get_predictions(local_weight_path):
    # Load your model
    model = YOLO(local_weight_path)

    # Predict on an image
    results = model(image_path)
    class_names = model.names

    return results, class_names


def upload_predictions(api: sly.Api, results, class_names, image_path):
    labels = []
    obj_classes = []
    for name in class_names.values():
        obj_classes.append(sly.ObjClass(name, sly.Rectangle))
    project_meta = sly.ProjectMeta(obj_classes=obj_classes)

    # Process results list
    for result in results:
        boxes = result.boxes.cpu().numpy()  # bbox outputs
        for box in boxes:
            class_name = class_names[int(box.cls[0])]
            obj_class = project_meta.get_obj_class(class_name)
            left, top, right, bottom = box.xyxy[0].astype(int)
            bbox = sly.Rectangle(top, left, bottom, right)
            labels.append(sly.Label(bbox, obj_class))

    new_project = api.project.create(WORKSPACE_ID, "predictions", change_name_if_conflict=True)
    new_dataset = api.dataset.create(new_project.id, "dataset")
    api.project.update_meta(new_project.id, project_meta.to_json())
    image_info = api.image.upload_path(new_dataset.id, "image.jpeg", image_path)
    ann = sly.Annotation((image_info.height, image_info.width), labels=labels)
    api.annotation.upload_ann(image_info.id, ann)

    return new_project


if __name__ == "__main__":
    api = sly.Api()
    result_folder, best_weight = train_model(api)
    sly.logger.info("Training completed")
    sly.logger.info(
        "The weights of trained model, predictions visualization and other training artifacts can be found in the following Team Files folder:"
    )
    best_weight_local = download_weight(api, best_weight)
    results, class_names = get_predictions(best_weight_local)
    new_project = upload_predictions(api, results, class_names, image_path)
    sly.logger.info(f"New project created. ID: {new_project.id}, name: {new_project.name}")
