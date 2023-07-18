import os
from pathlib import Path
from time import sleep

import cv2
import requests
import supervisely as sly
import torch
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv(os.path.expanduser("~/supervisely.env"))
load_dotenv("local.env")

api = sly.Api()

# get env variables
GLOBAL_TIMEOUT = 1  # seconds
AGENT_ID = 230  # agent id to run training on
APP_NAME = "supervisely-ecosystem/yolov8/train"
PROJECT_ID = sly.env.project_id()
DATASET_ID = sly.env.dataset_id()
TEAM_ID = sly.env.team_id()
WORKSPACE_ID = sly.env.workspace_id()
DATA_DIR = sly.app.get_data_dir()
task_type = "object detection"  # you can choose "instance segmentation" or "pose estimation"


############################################################
##################### PART 1: TRAINING #####################
############################################################

module_id = api.app.get_ecosystem_module_id(APP_NAME)
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
        "task_type": task_type,
        "model": "YOLOv8n-det",
        "train_mode": "finetune",  # finetune / scratch
        "n_epochs": 100,
        "patience": 50,
        "batch_size": 16,
        "input_image_size": 640,
        "optimizer": "AdamW",  # AdamW, Adam, SGD, RMSProp
        "n_workers": 8,
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "amp": True,
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
    },  # 📗 train paramaters
    timeout=10e6,
)

team_files_folder = Path("/yolov8_train") / task_type / project_name / str(task_id)
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

sly.logger.info("Training completed")
sly.logger.info(
    f"The weights of trained model and other artifacts uploaded in Team Files: {str(team_files_folder)}"
)


############################################################
############## PART 2: Download model weight ###############
############################################################

# download best weight from Supervisely Team Files
weight_name = os.path.basename(best)
weight_dir = os.path.join(DATA_DIR, "weights")
local_weight_path = os.path.join(weight_dir, weight_name)
if sly.fs.dir_exists(weight_dir):
    sly.fs.remove_dir(weight_dir)

api.file.download(TEAM_ID, best, local_weight_path)
sly.logger.info(f"Model weight downloaded to {local_weight_path}")


############################################################
################ PART 3: Perform inference #################
############################################################

# Load your model
model = YOLO(local_weight_path)


# define device
device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

# load image
input_image = sly.image.read(image_path)
input_image = input_image[:, :, ::-1]
input_height, input_width = input_image.shape[:2]

# Predict on an image
results = model(
    source=input_image,
    conf=0.25,
    iou=0.7,
    half=False,
    device=device,
    max_det=300,
    agnostic_nms=False,
)

# visualize predictions
predictions_plotted = results[0].plot()
cv2.imwrite(os.path.join(DATA_DIR, "predictions.jpg"), predictions_plotted)


############################################################
############## PART 4: Upload to Supervisely ###############
############################################################

# Get class names
class_names = model.names

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

# Create project, dataset and update project meta
project = api.project.create(WORKSPACE_ID, "predictions", change_name_if_conflict=True)
dataset = api.dataset.create(project.id, "dataset")
api.project.update_meta(project.id, project_meta.to_json())

# Upload image to Supervisely
image_info = api.image.upload_path(dataset.id, "image.jpeg", image_path)

# Create annotation for image and upload it
ann = sly.Annotation((image_info.height, image_info.width), labels=labels)
api.annotation.upload_ann(image_info.id, ann)

sly.logger.info(f"Annotated image (ID:{image_info.id}) uploaded to project (ID:{project.id})")
