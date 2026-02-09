import onnx
import onnxsim
import torch
import torch.nn as nn
from models.yolo import Model
from models.experimental import End2End
from models.common import *
from utils.add_nms import RegisterNMS
from torch.serialization import add_safe_globals
add_safe_globals([Model, nn.Sequential])

import torch._dynamo
torch._dynamo.config.suppress_errors = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = 'yolov7-tiny.pt'
ONNX_NAME = "yolov7-tiny.onnx"
IMG_SIZE = (640, 640)
BATCH_SIZE = 1
conv_cls = Conv

TOPK = 100
CONF_THRES = 0.35
IOU_THRES = 0.65

if __name__ == '__main__':
    ckpt = torch.load(MODEL_NAME, map_location=DEVICE, weights_only=False)
    model = ckpt["ema" if ckpt.get("ema") else "model"]
    model = model.float().fuse().eval().to(DEVICE)

    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None
        elif Conv and isinstance(m, Conv):
            m._non_persistent_buffers_set = set()

    model.eval()
    labels = model.names

    dummy_input = torch.zeros(BATCH_SIZE, 3, *IMG_SIZE).to(DEVICE)

    dynamic_axes = {'images': {0: 'batch'} }

    output_axes = {
        'num_dets': {0: 'batch'},
        'det_boxes': {0: 'batch'},
        'det_scores': {0: 'batch'},
        'det_classes': {0: 'batch'},
    }

    dynamic_axes.update(output_axes)

    model = End2End(
        model, 
        max_obj=TOPK,
        iou_thres=IOU_THRES,
        score_thres=CONF_THRES,
        max_wh=None,
        device=DEVICE,
        n_classes=len(labels)
    ).eval()

    output_names = ['num_dets', 'det_boxes', 'det_scores', 'det_classes']
    shapes = [BATCH_SIZE, 1, BATCH_SIZE, TOPK, 4, BATCH_SIZE, TOPK, BATCH_SIZE, TOPK]

    torch.onnx.export(
        model, 
        dummy_input, 
        ONNX_NAME, 
        verbose=False, 
        opset_version=13, 
        input_names=['images'],            
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
        do_constant_folding=True
    )

    onnx_model = onnx.load(ONNX_NAME)
    onnx.checker.check_model(onnx_model)

    for i in onnx_model.graph.output:
        for j in i.type.tensor_type.shape.dim:
            j.dim_param = str(shapes.pop(0))

    onnx_model, check = onnxsim.simplify(onnx_model)
    assert check, 'assert check failed'

    print(onnx.printer.to_text(onnx_model.graph))
    onnx.save(onnx_model, ONNX_NAME)