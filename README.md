# /License_plates_recognition
Vehicle license plate recognition with Yolo v5
# Clone the repository
```
git clone https://github.com/conggalam12//License_plates_recognition.git
```
# Install requirement
```
pip install -r requirements.txt
```
# Download and setup weights
```
mkdir weights
cd weights
wget https://github.com/conggalam12/Yolo-v5/releases/download/weights/bienso.pt
```
# How to use
```
python detect.py --source [path_image] --weights [path to weight] 
```
The result save in folder run/detect 
# Change folder save
```
python detect.py --source [path_image] --weights [path to weight] --project [path to folder]
```
# Example 
![detect](https://github.com/conggalam12/Yolo-v5/blob/main/runs/detect/exp/5.jpg)

![detect](https://github.com/conggalam12/Yolo-v5/blob/main/runs/detect/exp2/22.jpg)
