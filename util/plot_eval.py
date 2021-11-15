from pathlib import Path
import os

import torch
from PIL import Image
import torchvision.transforms as T
from plot_utils import plot_image_results
from box_ops import rescale_bboxes

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def filter_bboxes_from_outputs(outputs,
                               im,
                               threshold=0.7):

  # keep only predictions with confidence above threshold
  probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
  keep = probas.max(-1).values > threshold

  probas_to_keep = probas[keep]

  # convert boxes from [0; 1] to image scales
  bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep],
                                 im.size)

  return probas_to_keep, bboxes_scaled

def run_visual_validation_workflow(my_image, my_model):
  # mean-std normalize the input image (batch-size: 1)
  img = transform(my_image).unsqueeze(0)

  # propagate through the model
  outputs = my_model(img)

  for threshold in [0.7, 0.35]:

    probas_to_keep, bboxes_scaled = filter_bboxes_from_outputs(outputs,
                                                               img,
                                                               threshold=threshold)

    plot_image_results(my_image,
                       probas_to_keep,
                       bboxes_scaled)


def run_visual_validation(args):
    # Get the model ready
    model = torch.hub.load('facebookresearch/detr',
                           'detr_resnet50',
                           pretrained=False,
                           num_classes=args.num_classes)
    checkpoint = torch.load(Path(args.output_dir) / 'checkpoint.pth',
                            map_location='cpu')

    model.load_state_dict(checkpoint['model'],
                        strict=False)
    model.eval()

    # Iterate over the validation folder and run the model. For visual feedback.
    val_folder = Path(args.coco_path) / 'val'
    for img_name in os.listdir(val_folder):
        im = Image.open(Path(val_folder) / img_name)
        run_visual_validation_workflow(im, model)