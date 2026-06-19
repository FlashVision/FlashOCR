"""
Recognize Text in an Image
============================
Load a trained model and recognize text in an image.
"""
from flashocr import Predictor

predictor = Predictor(
    model_path="workspace/my_model/best.pth",
    device="cuda",
)

text, confidence = predictor.recognize_image("test_image.jpg")
print(f"Recognized: '{text}' (confidence: {confidence:.2f})")
