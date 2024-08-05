# License_plates_recognition
Vehicle license plate recognition with Yolo v5
# Clone the repository
```
git clone https://github.com/conggalam12/License_plates_recognition.git
```
# Install requirement
```
pip install -r requirements.txt
```
# Download and setup weights
```
mkdir weights
cd weights
wget https://github.com/conggalam12/License_plates_recognition/releases/download/weights/model_detector.pt
wget https://github.com/conggalam12/License_plates_recognition/releases/download/weights/model_ocr.pt
```
# How to use
```
python demo_fix.py  
```

# Example 
![detect](https://github.com/conggalam12/License_plates_recognition/blob/main/result_image/demo.jpg)

![detect](https://github.com/conggalam12/License_plates_recognition/blob/main/result_image/demo2.jpg)

![detect](https://github.com/conggalam12/License_plates_recognition/blob/main/result_image/demo3.jpg)
