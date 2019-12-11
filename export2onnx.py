from models import DeepLabV3Plus
import cv2
import onnx
import torch

dummy_input = torch.rand([1, 3, 320, 320])
model = DeepLabV3Plus(32)
weights = torch.load('weights/best.pt', map_location='cpu')['model']
model.load_state_dict(weights)
model.eval()
with torch.no_grad():
    model(dummy_input)
torch.onnx.export(model, dummy_input, 'best.onnx', opset_version=11)
onnxmodel = onnx.load('best.onnx')
onnx.checker.check_model(onnxmodel)
net = cv2.dnn.readNetFromONNX('best.onnx')
net.setInput(dummy_input.numpy())
net.forward()
