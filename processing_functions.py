import cv2
import numpy as np
def postprocess(frame, outs, confThreshold,nmsThreshold, classes, mask = False):  
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
        if mask :
          if w/h < 2 and w/h > 0.5:
            frame = drawPred(frame, classIds[i], confidences[i], x, y, x+w, y+h, classes)
        else:
          frame = drawPred(frame, classIds[i], confidences[i], x, y, x+w, y+h, classes)
    return frame



# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def drawPred(frame, classId, conf, left, top, right, bottom, classes):
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

def display_count(frame, height, width, count_classes):
  count_north,count_south = count_classes
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

def track_bbox(boxes_previous,box, confidence, predicted_class,k,count_classes):
    count_north, count_south = count_classes
    add = 1
    for i in range(len(boxes_previous)):
      bbox_previous, prev_proba, prev_class = boxes_previous[i]
      iou_calc = bbox_iou(box, bbox_previous)
      if iou_calc > 0.3:
        if predicted_class != prev_class :
          count = None
          count = count_south if box[0] > 320 else count_north
          if predicted_class in count.keys(): 
            count[predicted_class] += 1
            count[prev_class] -= 1
        add = 0
    if add == 1:
      count = None
      count = count_south if box[0] > 320 else count_north
      if predicted_class in count.keys(): count[predicted_class] += 1
    return [count_north,count_south]
	  
def postprocess2(frames, outs, confThreshold, nmsThreshold, prev_boxes, classes, mask =False, count_classes=False):
  height,width,channels = frames[0].shape
  output_frames = []; acc_boxes = []
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
      if mask:
        if w/h < 2 and w/h > 0.5:
          acc_boxes.append([boxes[i], confidences[i],classes[classIds[i]]])
          count_classes = track_bbox(prev_boxes, boxes[i], confidences[i],classes[classIds[i]],k, count_classes)
          frame = drawPred(frames[k],classIds[i], confidences[i], x, y, x+w, y+h, classes)
      else:
        frame = drawPred(frames[k],classIds[i], confidences[i], x, y, x+w, y+h, classes)
    prev_boxes = acc_boxes
    acc_boxes = []
    if count_classes:
      frame = display_count(frame, height, width, count_classes)
    output_frames.append(frame)
  return output_frames, prev_boxes, count_classes

def yolo_predict_image(img, net):
	inpWidth = 416       #Width of network's input image
	inpHeight = 416      #Height of network's input image
	# Create a 4D blob from a frame.
	blob = cv2.dnn.blobFromImage(img, 1/255., (inpWidth, inpHeight), [0,0,0], swapRB=True, crop=False)
	# Sets the input to the network
	net.setInput(blob)
	# Runs the forward pass to get output of the output layers
	outs = net.forward(getOutputsNames(net))
	return outs
	
def yolo_init():
	# Load names of classes
	classesFile = "coco.names"
	classes = None
	with open(classesFile, 'rt') as f:
		classes = f.read().rstrip('\n').split('\n')
	# Give the configuration and weight files for the model and load the network using them.
	modelConfiguration = "yolov3.cfg"
	modelWeights = "yolov3.weights"
	net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
	return net,classes

def images_array(fname,time_in_mins=1,skip=1, mask =False):
  ''' fname : file name in the working directory or the file path
      time_in_mins : video length to be considered for prediction
      mask : masking feature to predict in only a specific region of image
  '''
  frames = []
  imgs = []
  cap = cv2.VideoCapture(fname)
  Total_frames = cap.get(7)
  fps = round(cap.get(cv2.CAP_PROP_FPS))
  f_no = fps*60*time_in_mins
  f_no = int(f_no) if f_no < Total_frames else int(Total_frames)
  for i in range(0,f_no, skip):
    cap.set(1, i)
    ret, frame = cap.read()
    frames.append(frame)
    if mask:
      img = mask_image(frame,mask)
      imgs.append(img)
    else:
      img = frame
      imgs.append(img)
  frames = np.array(frames)
  imgs = np.array(imgs)
  return imgs,frames

def predict_video(imgs,net,batch_size = 32):
  out_batches = []
  batches = len(imgs)//batch_size
  rem = len(imgs) % batch_size
  for i in range(batches+1):
    if i < batches :
      img_batch = imgs[batch_size*i:batch_size*(i+1)]
    else:
      img_batch = imgs[batch_size*(i):batch_size*(i)+rem]
    blob = cv2.dnn.blobFromImages(img_batch, 1/255., (416, 416), [0,0,0], swapRB=True, crop=False)
    # Sets the input to the network
    net.setInput(blob)
    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))
    out_batches.append(outs)
  return out_batches, batches, rem

def output_frames(frames, out_batches, batch_size, batches, rem, classes, count_classes = False, confThreshold = 0.4, nmsThreshold = 0.4 ):
  count_north = {'car':0, 'bus': 0, 'truck': 0}
  count_south = {'car':0, 'bus': 0, 'truck': 0}
  count_classes = [count_north, count_south]
  prev_boxes = []
  out_frame_batches = []
  for i in range(batches+1):
    if i < batches :
      frame_batch = frames[batch_size*i:batch_size*(i+1)]
    else:
      frame_batch = frames[batch_size*(i):batch_size*(i)+rem]
    outs = out_batches[i]
    output_frames, prev_boxes, count_classes = postprocess2(frame_batch, outs, confThreshold, nmsThreshold, prev_boxes,classes,mask=True, count_classes=count_classes)
    out_frame_batches.append(output_frames)
  return out_frame_batches

def create_video(out_frame_batches,video_loc,fps):
  height, width, channels = out_frame_batches[0][0].shape
  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  out = cv2.VideoWriter(video_loc,fourcc, fps, (width,height))
  for output_frames in out_frame_batches:
    for i in range(len(output_frames)):
      out.write(output_frames[i])
  out.release()

def mask_image(img,refPt):
    pts = np.array(refPt)
    pts = pts*2
    #print(pts)
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(img, img, mask=mask)
    return dst