import copy
import os
import sys 
import tempfile
from dds_cloudapi_sdk import Config, Client
from dds_cloudapi_sdk.tasks.v2_task import V2Task
import argparse
import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image
import gc

# 1. 获取当前脚本的绝对路径
current_file_path = os.path.abspath(__file__)
# 2. 获取当前脚本所在的目录 (.../level2/)
current_dir = os.path.dirname(current_file_path)
# 3. 获取上级目录 (.../level1/)
project_root = os.path.dirname(current_dir)
# 4. 添加到 sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from utils.common_utils import CommonUtils
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images

# Setup environment
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

class DDSGroundingDinoPredictor:
    """
    Wrapper for using DDS Cloud API (GroundingDINO 1.5) for detection.
    """
    def __init__(self, api_token: str, model_name="GroundingDino-1.5-Pro", device="cuda"):
        """
        Initialize the DDS Cloud API predictor.
        Args:
            api_token (str): Your DDS API token.
            model_name (str): The model to use (e.g., "GroundingDino-1.5-Pro").
            device (str): Device to move output tensors to.
        """
        self.model_name = model_name
        self.config = Config(api_token)
        self.client = Client(self.config)
        self.device = device
        print(f"[DDSGroundingDinoPredictor] Initialized with model: {self.model_name}")

    def predict(
        self,
        image: "PIL.Image.Image",
        text_prompts: str,
        box_threshold=0.3,
        text_threshold=0.3, # Note: text_threshold is aliased to iou_threshold here
    ):
        """
        Perform object detection using the DDS Cloud API.
        Args:
            image (PIL.Image.Image): Input RGB image.
            text_prompts (str): Text prompt describing target objects.
            box_threshold (float): Confidence threshold for box selection.
            text_threshold (float): Mapped to iou_threshold for the API.
        Returns:
            Tuple[Tensor, List[str]]: Bounding boxes and matched class labels.
        """
        print("[DDS] Preparing image for upload...")
        # 1. Convert PIL Image to np array (BGR for cv2)
        image_np_rgb = np.array(image)
        image_np_bgr = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR)

        # 2. Save to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmpfile:
            temp_filename = tmpfile.name
            cv2.imwrite(temp_filename, image_np_bgr)

        try:
            # 3. Upload file
            print(f"[DDS] Uploading {temp_filename}...")
            image_url = self.client.upload_file(temp_filename)
            print(f"[DDS] Image uploaded, URL: {image_url}")

            # 4. Define and run the task
            task = V2Task(
                api_path="/v2/task/grounding_dino/detection",
                api_body={
                    "model": self.model_name,
                    "image": image_url,
                    "prompt": {
                        "type": "text",
                        "text": text_prompts
                    },
                    "targets": ["bbox"],
                    "bbox_threshold": box_threshold,
                    "iou_threshold": text_threshold, # Using text_threshold as IOU
                }
            )
            print("[DDS] Running detection task... (This may take a moment)")
            self.client.run_task(task)
            result = task.result
            print("[DDS] Task complete, processing results.")

            # 5. Process results
            objects = result.get("objects", [])
            input_boxes = []
            labels = []

            if not objects:
                print("[DDS] No objects found.")
                return torch.empty((0, 4), device=self.device), []

            for obj in objects:
                input_boxes.append(obj["bbox"])
                labels.append(obj["category"].lower().strip())
            
            # Convert to tensor and move to device
            boxes_tensor = torch.tensor(np.array(input_boxes).reshape(-1, 4), dtype=torch.float32).to(self.device)
            return boxes_tensor, labels

        finally:
            # 6. Clean up temporary file
            os.remove(temp_filename)
            print(f"[DDS] Cleaned up temp file {temp_filename}.")

class SAM2ImageSegmentor:
    """
    Wrapper class for SAM2-based segmentation given bounding boxes.
    """

    def __init__(self, sam_model_cfg: str, sam_model_ckpt: str, device="cuda"):
        """
        Initialize the SAM2 image segmentor.
        Args:
            sam_model_cfg (str): Path to the SAM2 config file.
            sam_model_ckpt (str): Path to the SAM2 checkpoint file.
            device (str): Device to load the model on ('cuda' or 'cpu').
        """
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = device
        sam_model = build_sam2(sam_model_cfg, sam_model_ckpt, device=device)
        self.predictor = SAM2ImagePredictor(sam_model)

    def set_image(self, image: np.ndarray):
        """
        Set the input image for segmentation.
        Args:
            image (np.ndarray): RGB image array with shape (H, W, 3).
        """
        self.predictor.set_image(image)

    def predict_masks_from_boxes(self, boxes: torch.Tensor):
        """
        Predict segmentation masks from given bounding boxes.
        Args:
            boxes (torch.Tensor): Bounding boxes as (N, 4) tensor.
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - masks: Binary masks per box, shape (N, H, W)
                - scores: Confidence scores for each mask
                - logits: Raw logits from the model
        """
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )

        # Normalize shape to (N, H, W)
        if masks.ndim == 2:
            masks = masks[None]
            scores = scores[None]
            logits = logits[None]
        elif masks.ndim == 4:
            masks = masks.squeeze(1)

        return masks, scores, logits


class IncrementalObjectTracker:
    def __init__(
        self,
        api_token: str,
        grounding_model_name: str = "GroundingDino-1.5-Pro",
        sam2_model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
        sam2_ckpt_path="./checkpoints/sam2.1_hiera_large.pt",
        device="cuda",
        prompt_text="car.",
        detection_interval=20,
    ):
        self.device = device
        self.detection_interval = detection_interval
        self.prompt_text = prompt_text

        # Load models
        self.grounding_predictor = DDSGroundingDinoPredictor(
            api_token=api_token, model_name=grounding_model_name, device=device
        )
        self.sam2_segmentor = SAM2ImageSegmentor(
            sam_model_cfg=sam2_model_cfg,
            sam_model_ckpt=sam2_ckpt_path,
            device=device,
        )
        self.video_predictor = build_sam2_video_predictor(
            sam2_model_cfg, sam2_ckpt_path
        )

        # Initialize state
        self.inference_state = self.video_predictor.init_state()
        self.inference_state["images"] = torch.empty((0, 3, 1024, 1024), device=device)
        self.inference_state["video_height"] = None
        self.inference_state["video_width"] = None

        self.total_frames = 0
        self.objects_count = 0
        self.frame_cache_limit = 40

        self.last_mask_dict = MaskDictionaryModel()
        self.track_dict = MaskDictionaryModel()

        # ✅ new fields for memory reset without losing masks
        self.next_anchor_masks = None
        self.need_reset_next_frame = False

    # ============================================================
    # ✅ add_frame: handles cache and re-anchor logic
    # ============================================================
    def add_frame(self, image_np):
        """
        改动目标：
            - 不做重锚定（不重新 add_new_mask）
            - 仅在缓存过多时清理 inference_state 中的旧帧数据
            - 保持跟踪连续性
        """
        frame_idx = self.video_predictor.add_new_frame(self.inference_state, image_np)

        # 获取当前缓存帧数
        frames_cached = self.inference_state["images"].shape[0]

        if frames_cached > self.frame_cache_limit:
            keep_n = self.frame_cache_limit // 2
            for key in ["images", "features", "embeddings"]:
                if key in self.inference_state and isinstance(self.inference_state[key], torch.Tensor):
                    if self.inference_state[key].shape[0] > keep_n:
                        self.inference_state[key] = self.inference_state[key][-keep_n:].clone()

            gc.collect()
            torch.cuda.empty_cache()
            print(f"[Memory Prune] Cached frames pruned to last {keep_n} entries")

        return frame_idx



    # ============================================================
    # ✅ modified add_image -> uses add_frame
    # ============================================================
    def add_image(self, image_np: np.ndarray):

        img_pil = Image.fromarray(image_np)

        # 初始化视频尺寸
        if self.inference_state["video_height"] is None:
            self.inference_state["video_height"], self.inference_state["video_width"] = image_np.shape[:2]

        # --------------------------------------------------------
        # A) 检测帧
        # --------------------------------------------------------
        if self.total_frames % self.detection_interval == 0:
            boxes, labels = self.grounding_predictor.predict(img_pil, self.prompt_text)

            if boxes.shape[0] == 0:
                # 无检测结果 → 直接通过传播
                frame_idx = self.add_frame(image_np)
            else:
                with torch.no_grad():
                    self.sam2_segmentor.set_image(image_np)
                    masks, scores, logits = self.sam2_segmentor.predict_masks_from_boxes(boxes)

                mask_dict = MaskDictionaryModel(
                    promote_type="mask",
                    mask_name=f"mask_{self.total_frames:05d}.png"
                )
                mask_dict.add_new_frame_annotation(
                    mask_list=torch.tensor(masks).to(self.device),
                    box_list=torch.tensor(boxes),
                    label_list=labels,
                )

                self.objects_count = mask_dict.update_masks(
                    tracking_annotation_dict=self.last_mask_dict,
                    iou_threshold=0.3,
                    objects_count=self.objects_count,
                )

                frame_idx = self.add_frame(image_np)
                self.video_predictor.reset_state(self.inference_state)

                for object_id, object_info in mask_dict.labels.items():
                    frame_idx, _, _ = self.video_predictor.add_new_mask(
                        self.inference_state,
                        frame_idx,
                        object_id,
                        object_info.mask,
                    )

                # 不重锚定，不 reset_state，直接保留现有跟踪状态
                self.track_dict = copy.deepcopy(mask_dict)
                self.last_mask_dict = mask_dict

        # --------------------------------------------------------
        # B) 非检测帧 → 正常增量跟踪
        # --------------------------------------------------------
        else:
            frame_idx = self.add_frame(image_np)

        # --------------------------------------------------------
        # C) 推理传播当前帧掩码
        # --------------------------------------------------------
        with torch.no_grad():
            frame_idx, obj_ids, video_res_masks = self.video_predictor.infer_single_frame(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
            )

        frame_masks = MaskDictionaryModel()
        for i, obj_id in enumerate(obj_ids):
            out_mask = video_res_masks[i] > 0.0
            object_info = ObjectInfo(
                instance_id=obj_id,
                mask=out_mask[0],
                class_name=self.track_dict.get_target_class_name(obj_id),
                logit=self.track_dict.get_target_logit(obj_id),
            )
            object_info.update_box()
            frame_masks.labels[obj_id] = object_info
            frame_masks.mask_name = f"mask_{self.total_frames + 1:05d}.png"
            frame_masks.mask_height = out_mask.shape[-2]
            frame_masks.mask_width = out_mask.shape[-1]

        self.last_mask_dict = copy.deepcopy(frame_masks)

        # 生成可视化帧
        H, W = image_np.shape[:2]
        mask_img = torch.zeros((H, W), dtype=torch.int32)
        for obj_id, obj_info in self.last_mask_dict.labels.items():
            mask_img[obj_info.mask] = obj_id
        mask_array = mask_img.cpu().numpy()

        annotated_frame = self.visualize_frame_with_mask_and_metadata(
            image_np=image_np,
            mask_array=mask_array,
            json_metadata=self.last_mask_dict.to_dict(),
        )

        self.total_frames += 1
        if self.total_frames % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        print(f"[Tracker] Total processed frames: {self.total_frames}")
        return annotated_frame



    def set_prompt(self, new_prompt: str):
        """
        Dynamically update the GroundingDINO prompt and reset tracking state
        to force a new object detection.
        """
        self.prompt_text = new_prompt
        self.total_frames = 0  # Trigger immediate re-detection
        self.inference_state = self.video_predictor.init_state()
        self.inference_state["images"] = torch.empty(
            (0, 3, 1024, 1024), device=self.device
        )
        self.inference_state["video_height"] = None
        self.inference_state["video_width"] = None

        print(f"[Prompt Updated] New prompt: '{new_prompt}'. Tracker state reset.")

    def save_current_state(self, output_dir, raw_image: np.ndarray = None):
            """
            Save the current mask, metadata, raw image, and annotated result.
            Modified to save masks as PNG instead of NPY.
            """
            mask_data_dir = os.path.join(output_dir, "mask_data")
            json_data_dir = os.path.join(output_dir, "json_data")
            image_data_dir = os.path.join(output_dir, "images")
            vis_data_dir = os.path.join(output_dir, "result")

            os.makedirs(mask_data_dir, exist_ok=True)
            os.makedirs(json_data_dir, exist_ok=True)
            os.makedirs(image_data_dir, exist_ok=True)
            os.makedirs(vis_data_dir, exist_ok=True)

            frame_masks = self.last_mask_dict

            # --- 修改点 1: 确保文件名后缀是 .png ---
            if not frame_masks.mask_name or not frame_masks.mask_name.endswith(".png"):
                # 如果之前的名字是以 .npy 结尾，替换掉，或者直接生成新的 .png 名字
                if frame_masks.mask_name and frame_masks.mask_name.endswith(".npy"):
                    frame_masks.mask_name = frame_masks.mask_name.replace(".npy", ".png")
                else:
                    frame_masks.mask_name = f"mask_{self.total_frames:05d}.png"

            base_name = f"image_{self.total_frames:05d}"

            # Save segmentation mask
            # 这里的 mask_img 是一张二维图，像素值等于 Object ID (1, 2, 3...)
            mask_img = torch.zeros(frame_masks.mask_height, frame_masks.mask_width)
            for obj_id, obj_info in frame_masks.labels.items():
                mask_img[obj_info.mask == True] = obj_id
            
            # --- 修改点 2: 使用 cv2.imwrite 保存为图片 ---
            # 注意：这里使用 uint16 格式，可以支持超过 255 个 ID 的情况。
            # 如果你确定 ID 不会超过 255，也可以改为 uint8。
            mask_numpy = mask_img.numpy().astype(np.uint16)
            
            save_path = os.path.join(mask_data_dir, frame_masks.mask_name)
            cv2.imwrite(save_path, mask_numpy)
            # -------------------------------------------

            # Save metadata as JSON
            json_path = os.path.join(json_data_dir, base_name + ".json")
            frame_masks.to_json(json_path)

            # Save raw input image
            if raw_image is not None:
                image_bgr = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(image_data_dir, base_name + ".jpg"), image_bgr)

                # Save annotated image with mask, bounding boxes, and labels
                annotated_image = self.visualize_frame_with_mask_and_metadata(
                    image_np=raw_image,
                    mask_array=mask_numpy, # 这里可以直接传入刚才转换好的 numpy 数组
                    json_metadata=frame_masks.to_dict(),
                )
                annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    os.path.join(vis_data_dir, base_name + "_annotated.jpg"), annotated_bgr
                )
                print(
                    f"[Saved] {base_name}.jpg, mask (png) and annotated image saved."
                )

                
    def visualize_frame_with_mask_and_metadata(
        self,
        image_np: np.ndarray,
        mask_array: np.ndarray,
        json_metadata: dict,
    ):
        image = image_np.copy()
        H, W = image.shape[:2]

        # Step 1: Parse metadata and build object entries
        metadata_lookup = json_metadata.get("labels", {})

        all_object_ids = []
        all_object_boxes = []
        all_object_classes = []
        all_object_masks = []

        for obj_id_str, obj_info in metadata_lookup.items():
            instance_id = obj_info.get("instance_id")
            if instance_id is None or instance_id == 0:
                continue
            if instance_id not in np.unique(mask_array):
                continue

            object_mask = mask_array == instance_id
            all_object_ids.append(instance_id)
            x1 = obj_info.get("x1", 0)
            y1 = obj_info.get("y1", 0)
            x2 = obj_info.get("x2", 0)
            y2 = obj_info.get("y2", 0)
            all_object_boxes.append([x1, y1, x2, y2])
            all_object_classes.append(obj_info.get("class_name", "unknown"))
            all_object_masks.append(object_mask[None])  # Shape (1, H, W)

        # Step 2: Check if valid objects exist
        if len(all_object_ids) == 0:
            print("No valid object instances found in metadata.")
            return image

        # Step 3: Sort by instance ID
        paired = list(
            zip(all_object_ids, all_object_boxes, all_object_masks, all_object_classes)
        )
        paired.sort(key=lambda x: x[0])

        all_object_ids = [p[0] for p in paired]
        all_object_boxes = [p[1] for p in paired]
        all_object_masks = [p[2] for p in paired]
        all_object_classes = [p[3] for p in paired]

        # Step 4: Build detections
        all_object_masks = np.concatenate(all_object_masks, axis=0)
        detections = sv.Detections(
            xyxy=np.array(all_object_boxes),
            mask=all_object_masks,
            class_id=np.array(all_object_ids, dtype=np.int32),
        )
        labels = [
            f"{instance_id}: {class_name}"
            for instance_id, class_name in zip(all_object_ids, all_object_classes)
        ]

        # Step 5: Annotate image
        annotated_frame = image.copy()
        mask_annotator = sv.MaskAnnotator()
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        annotated_frame = mask_annotator.annotate(annotated_frame, detections)
        annotated_frame = box_annotator.annotate(annotated_frame, detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)

        return annotated_frame

# ... (IncrementalObjectTracker 类的定义结束)