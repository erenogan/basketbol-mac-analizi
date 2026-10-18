import torch, cv2, ultralytics
print("torch:", torch.__version__)
print("cv2:", cv2.__version__)
print("gpu:", torch.cuda.is_available())
print("kart:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "YOK")