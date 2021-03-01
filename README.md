## Object detection and Tracking using YoloV3 in OpenCV

* OpenCV is used for video pre-processing.
* Region of interest is chosen to detect the objects only when seen in the particular region.
* Objects are detected by calling YoloV3 model into opencv.
* Tracking is done by calculating Intersection over union for predicted bounding boxes of consecutive frames
* The locations of bounding boxes are used to predict the direction of motion of vehicles

### Demo Colab link:

* https://colab.research.google.com/github/dheerajreddy2020/OpenCV_Yolo_Object_detection_tracking/blob/master/Yolo_predict_demo.ipynb
