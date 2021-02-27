import cv2
import numpy as np
def postprocess(frame, outs):  
    height,width,channels = frame.shape
    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                x_c = int(detection[0] * width)
                y_c = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(x_c - w / 2)
                y = int(y_c - h / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        x,y,w,h = box
        if w/h < 2 and w/h > 0.5:
          frame = drawPred(frame, classIds[i], confidences[i], x, y, x+w, y+h)
    cv2_imshow(frame)


# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def drawPred(frame, classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255),2)
    label = '%.2f' % conf
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
        #Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        top = max(top, labelSize[1])
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
    return frame

def display_count(frame):
  i = 3
  cv2.putText(frame, 'North Direction',(5, 10+i*5), cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * height, (255,128,0), 2)
  for key,value in count_north.items():
    i += 3
    cv2.putText(frame, key +' : ', (5, 10+i*5), cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * height, (0,128,255), 2)
    cv2.putText(frame, str(value), (100, 10+i*5), cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * height, (255,255,255), 2)
  i = 3
  cv2.putText(frame, 'South Direction',(width - 150, 10+i*5), cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * height, (255,128,0), 2)
  for key,value in count_south.items():
    i += 3
    cv2.putText(frame, key +' : ', (width - 150, 10+i*5), cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * height, (0,128,255), 2)
    cv2.putText(frame, str(value), (width - 55, 10+i*5), cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * height, (255,255,255), 2)
  return frame

def interval_overlap(interval_a, interval_b):
	x1, x2 = interval_a
	x3, x4 = interval_b
	if x3 < x1:
		if x4 < x1:
			return 0
		else:
			return min(x2,x4) - x1
	else:
		if x2 < x3:
			 return 0
		else:
			return min(x2,x4) - x3

def bbox_iou(box1, box2):
	intersect_w = interval_overlap([box1[0], box1[0]+box1[2]], [box2[0], box2[0]+box2[2]])
	intersect_h = interval_overlap([box1[1], box1[1]+box1[3]], [box2[1], box2[1]+box2[3]])
	intersect = intersect_w * intersect_h
	w1, h1 = box1[2], box1[3]
	w2, h2 = box2[2], box2[3]
	union = w1*h1 + w2*h2 - intersect
	return float(intersect) / union

def track_bbox(boxes_previous,box, confidence, predicted_class,k):
    add = 1
    for i in range(len(boxes_previous)):
      bbox_previous, prev_proba, prev_class = boxes_previous[i]
      iou_calc = bbox_iou(box, bbox_previous)
      if iou_calc > 0.4:
        if predicted_class != prev_class :
          count = None
          count = count_south if box[0] > 320 else count_north
          if predicted_class in count.keys(): count[predicted_class] += 1
          count[prev_class] -= 1
        add = 0
    if add == 1:
      count = None
      count = count_south if box[0] > 320 else count_north
      if predicted_class in count.keys(): count[predicted_class] += 1
	  
def postprocess2(frames, outs, count_classes = False):
  output_frames = []
  acc_boxes = []; prev_boxes = []
  for k in range(len(frames)):  
    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out[k]:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                x_c = int(detection[0] * width)
                y_c = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(x_c - w / 2)
                y = int(y_c - h / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        x,y,w,h = boxes[i]
        #print(boxes[i])
        if w/h < 2 and w/h > 0.5:
          acc_boxes.append([boxes[i], confidences[i],classes[classIds[i]]])
          track_bbox(prev_boxes, boxes[i], confidences[i],classes[classIds[i]],k)
          frame = drawPred(frames[k],classIds[i], confidences[i], x, y, x+w, y+h)
    prev_boxes = acc_boxes
    acc_boxes = []
    if count_classes:
      frame = display_count(frame)
    output_frames.append(frame)
  return output_frames