import os
import cv2
import time
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf

from utils.app_utils import FPS, WebcamVideoStream
from multiprocessing import Queue, Pool
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from time import sleep

CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph1.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 1

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def detect_objects(image_np, sess, detection_graph, im_height, im_width):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    # print(image_np.size)
    # im_height , im_width = image_np.size
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})


    persons = []
    for i, clas in enumerate(classes):
        for j, c in enumerate(clas):
            if c == 1 and scores[i,j] > 0.5:
                box = boxes[i][j]
                ymin = (box[0] * im_height).astype(np.int)
                xmin = (box[1] * im_width).astype(np.int)
                ymax = (box[2] * im_height).astype(np.int)
                xmax = (box[3] * im_width).astype(np.int)
                print(ymin, xmin, ymax, xmax)
                person_img = image_np[ymin:ymax, xmin:xmax]
                persons.append(person_img)


    # for i, scr in enumerate(scores):
    #     for j, s in enumerate(scr):
    #             if s > 0.5:
    #                 if(classes[i][j] == 1):
    #                     box = boxes[i][j]
    #                     # print(box)
    #                     ymin = (box[0] * im_height).astype(np.int)
    #                     xmin = (box[1] * im_width).astype(np.int)
    #                     ymax = (box[2] * im_height).astype(np.int)
    #                     xmax = (box[3] * im_width).astype(np.int)
    #                     print(ymin, xmin, ymax, xmax)
    #                     person_img = image_np[ymin:ymax, xmin:xmax]
    #                     # plt.figure(figsize=image_np.size)
    #                     # plt.imshow(dog)
    #                     # detect_haar(person_img)
    #                     persons.append(person_img)
    if(len(persons) > 0):
        return persons



    # i = 0
    # for box in boxes:
    #     print(box)
    #     print(classes[i])
    #     print(scores[i])
    #     i = i +1
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    return image_np


def worker(input_q, output_q):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)
    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h , w , x = frame_rgb.shape
        persons = detect_objects(frame_rgb, sess, detection_graph, h, w)
        if isinstance(persons, list):
            for p in persons:
                output_q.put(p)
        else:
            output_q.put(persons)
        # output_q.put(detect_objects(frame_rgb, sess, detection_graph, h, w))
        # output_q.put(frame_rgb)
    fps.stop()
    sess.close()

def detect_haar(person_img,  minSize_= (50,60), scaleFactor_=1.6, minNeighbors_=12, cascadepath = 'cascade.xml'):
    cascade = cv2.CascadeClassifier(cascadepath)
    # video_capture = cv2.VideoCapture(videofile)
    # ret, frame = video_capture.read()
    gray = cv2.cvtColor(person_img, cv2.COLOR_BGR2GRAY)
    detects = cascade.detectMultiScale(gray,
            scaleFactor= scaleFactor_,
            minNeighbors=minNeighbors_,
            minSize= minSize_, #(14,28), #(60, 20),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
    #draw_detections(frame, detects)
    draw_detectionsFull(person_img, detects)
    return person_img

def draw_detectionsFull(img, rects, thickness = 3, color = (255,0,0), subframe = False, X = 0, Y = 0):
    for x, y, w, h in rects:
        if(subframe):
            cv2.rectangle(img, (X+x+w, Y+y+h), (X+x+w-w, Y+y+h-h), color, thickness)
        else:
            cv2.rectangle(img, (x+w, y+h), (x+w-w, y+h-h), color, thickness)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=1, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=480, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=360, help='Height of the frames in the video stream.')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=4, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=10, help='Size of the queue.')
    args = parser.parse_args()

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)

    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)
    pool = Pool(args.num_workers, worker, (input_q, output_q))

    #video_capture = WebcamVideoStream(src=args.video_source,
    #                                 width=args.width,
    #                                  height=args.height).start()
    
    # video_capture = cv2.VideoCapture(0)
    file = 'file.mp4'
    video_capture = cv2.VideoCapture(file)
    fps = FPS().start()
    
    while True:
        if not video_capture.isOpened():
            print('Unable to load camera.')
            sleep(5)
            pass
    
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        
        input_q.put(frame)
        output_rgb = cv2.cvtColor(output_q.get(), cv2.COLOR_RGB2BGR)
        cv2.imshow('Video', output_rgb)
        fps.update()
       
        
        
       
    
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
        # Display the resulting frame
        #cv2.imshow('Video', frame)

    #while True:  # fps._numFrames < 120
    #    frame = video_capture.read()
        #input_q.put(frame)

        #t = time.time()

        #output_rgb = cv2.cvtColor(output_q.get(), cv2.COLOR_RGB2BGR)
        #cv2.imshow('Video', output_rgb)
        

        #print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    pool.terminate()
    video_capture.release()
    cv2.destroyAllWindows()
