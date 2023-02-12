from ast import arg
from mylib import config, thread
from mylib.detection import detect_people
from imutils.video import FPS
from scipy.spatial import distance as dist
import numpy as np
import argparse, imutils, cv2, os,time

def prog(videourl,liveurl):
	ap = argparse.ArgumentParser() 
	ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
	ap.add_argument("-d", "--display", type=int, default=1,
	help="whether or not output frame should be displayed")
	args = vars(ap.parse_args())

	labelsPath = os.path.sep.join(["yolo", "coco.names"])
	LABELS = open(labelsPath).read().strip().split("\n")

	weightsPath = os.path.sep.join(["yolo", "yolov3.weights"])
	configPath = os.path.sep.join(["yolo", "yolov3.cfg"])

	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

	ln = net.getLayerNames()
	ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

	if liveurl!="":
		print("Starting the live stream..")
		vs = cv2.VideoCapture(liveurl)
		if config.Thread:
			cap = thread.ThreadingClass(liveurl)
		time.sleep(2.0)
	else:
		print("Starting the video...")
		vs = cv2.VideoCapture(videourl)
		if config.Thread:
			cap = thread.ThreadingClass(videourl)

	writer = None
	fps = FPS().start()

	while True:
		if config.Thread:
			frame = cap.read()
		else:
			(grabbed, frame) = vs.read()
			if not grabbed:
				break

		frame = imutils.resize(frame, width=600)
		results = detect_people(frame, net, ln,personIdx=LABELS.index("person"))

		serious = set()

		if len(results) >= 2:
			centroids = np.array([r[2] for r in results])
			D = dist.cdist(centroids, centroids, metric="euclidean")

			for i in range(0, D.shape[0]):
				for j in range(i + 1, D.shape[1]):
					if D[i, j] < config.MIN_DISTANCE:
						serious.add(i)
						serious.add(j)

		for (i, (prob, bbox, centroid)) in enumerate(results):
			(startX, startY, endX, endY) = bbox
			(cX, cY) = centroid
			color = (0, 255, 0)
		
			if i in serious:
				color = (0, 0, 255)
		
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
			cv2.circle(frame, (cX, cY), 5, color, 2)

		text = "Total violations: {}".format(len(serious))
		cv2.putText(frame, text, (10, frame.shape[0] - 55),
			cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)
		if len(results) > 0:
			cv2.imshow("Social Distance Analysis Window", frame)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break
		fps.update()

		if args["output"] != "" and writer is None:
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, 25,
				(frame.shape[1], frame.shape[0]), True)

		if writer is not None:
			writer.write(frame)

	fps.stop()
	cv2.destroyAllWindows()