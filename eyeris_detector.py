import cv2
import numpy
from os.path import join
g = open("track.txt","w+")

def show_image_with_data(frame, blinks, landblinks, irises, window, err=None):
    """
    Helper function to draw points on eyes and display frame
    :param frame: image to draw on
    :param blinks: number of blinks
    :param window: for window dimension FW: added window for obtaining the window dimension
    :param irises: array of points with coordinates of irises
    :param err: for displaying current error in Lucas-Kanade tracker
    :return:
    """
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = window.get(cv2.CAP_PROP_FRAME_WIDTH) # float FW
    height = window.get(cv2.CAP_PROP_FRAME_HEIGHT) # float FW
    if err:
        cv2.putText(frame, str(err), (20, 450), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, 'blinks: ' + str(blinks), (int(0.9*width), int(0.97*height)), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    for w, h in irises:
        cv2.circle(frame, (w, h), 2, (0, 255, 0), 2)
    cv2.rectangle(frame,(int(0.01*width),int(0.0125*height)),(int(0.21*width),int(0.18*height)),(255,255,255),1) #takeoff rectangle FW
    cv2.rectangle(frame,(int(0.79*width),int(0.0125*height)),(int(0.99*width),int(0.18*height)),(255,255,255),1)   #landing rectangle FW
    cv2.imshow('Eyeris detector', frame)


class ImageSource:
    """
    Returns frames from camera
    """
    def __init__(self):
        self.capture = cv2.VideoCapture(0)

    def get_current_frame(self, gray=False):
        ret, frame = self.capture.read()
        frame = cv2.flip(frame, 1)  # 60fps
        if not gray:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def release(self):
        self.capture.release()


class CascadeClassifier:
    """
    This classifier is trained by default in OpenCV
    """
    def __init__(self, glasses=True):
        if glasses:
            self.eye_cascade = cv2.CascadeClassifier(join('haar', 'haarcascade_eye_tree_eyeglasses.xml'))
        else:
            self.eye_cascade = cv2.CascadeClassifier(join('haar', 'haarcascade_eye.xml'))

    def get_irises_location(self, frame_gray):
        eyes = self.eye_cascade.detectMultiScale(frame_gray, 1.3, 5)  # if not empty - eyes detected
        irises = []

        for (ex, ey, ew, eh) in eyes:
            iris_w = int(ex + float(ew / 2))
            iris_h = int(ey + float(eh / 2))
            irises.append([numpy.float32(iris_w), numpy.float32(iris_h)])

        return numpy.array(irises)


class LucasKanadeTracker:
    """
    Lucaas-Kanade tracker used for minimizing cpu usage and blinks counter
    """
    def __init__(self, blink_threshold=9):
        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.blink_threshold = blink_threshold

    def track(self, old_gray, gray, irises, blinks, blink_in_previous):
        lost_track = False
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, irises, None, **self.lk_params)
        if st[0][0] == 0 or st[1][0] == 0:  # lost track on eyes
            lost_track = True
            blink_in_previous = False
        elif err[0][0] > self.blink_threshold or err[1][0] > self.blink_threshold:  # high error rate in klt tracking
            lost_track = True
            if not blink_in_previous:
                blinks += 1
                blink_in_previous = True
        else:
            blink_in_previous = False
            irises = []
            for w, h in p1:
                irises.append([w, h])
            irises = numpy.array(irises)
        return irises, blinks, blink_in_previous, lost_track



class EyerisDetector:
    """
    Main class which use image source, classifier and tracker to estimate iris postion
    Algorithm used in detector is designed for one person (with two eyes)
    It can detect more than two eyes, but it tracks only two
    """
    def __init__(self, image_source, classifier, tracker):
        self.tracker = tracker
        self.classifier = classifier
        self.image_source = image_source
        self.irises = []
        self.blink_in_previous = False
        self.blinks = 0
        #self.takeoffblinks = 0
        self.landblinks = 0
    
    def run(self):
        counttf = []
        countld = []
        k = cv2.waitKey(30) & 0xff
        font = cv2.FONT_HERSHEY_SIMPLEX
        while k != 32:  # space
            frame = self.image_source.get_current_frame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if len(self.irises) >= 2:  # irises detected, track eyes
                track_result = self.tracker.track(old_gray, gray, self.irises, self.blinks, self.blink_in_previous)
                self.irises, self.blinks, self.blink_in_previous, lost_track = track_result
                width = self.image_source.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = self.image_source.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                takeoffwidth = [int(0.01*width),int(0.21*width)]
                takeoffheight = [int(0.0125*height),int(0.18*height)]
                landingwidth = [int(0.79*width),int(0.99*width)]
                landingheight = takeoffheight
                intakeoff = (takeoffwidth[1]>=int(self.irises[0][0])>=takeoffwidth[0] and takeoffheight[1]>=int(self.irises[0][1])>=takeoffheight[0]) or (takeoffwidth[1]>=int(self.irises[1][0])>=takeoffwidth[0] and takeoffheight[1]>=int(self.irises[1][1])>=takeoffheight[0])
                inlanding = (landingwidth[1]>=int(self.irises[0][0])>=landingwidth[0] and landingheight[1]>=int(self.irises[0][1])>=landingheight[0]) or (landingwidth[1]>=int(self.irises[1][0])>=landingwidth[0] and landingheight[1]>=int(self.irises[1][1])>=landingheight[0])
                
                if not(intakeoff or inlanding):
                    counttf[:] = []
                    countld[:] = []
                    g.write("waiting\r\n")
                
                if intakeoff:
                    countld[:] = []
                    cv2.rectangle(frame,(takeoffwidth[0],takeoffheight[0]),(takeoffwidth[1],takeoffheight[1]),(255,255,255),2)
                    counttf.append(self.irises)
                    waittime = 3 - int(len(counttf)/11)
                    if waittime > 0:
                        cv2.putText(frame, 'takeoff in: ' + str(waittime) + 'sec', (takeoffwidth[0],int(takeoffheight[1]/2)), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                        g.write("waiting\r\n")
                    if waittime <= 0:
                        #if waittime > -1:
                        g.write("takingoff\r\n")
                        cv2.putText(frame, 'taking off,please wait', (takeoffwidth[0],int(takeoffheight[1]/2)), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            
                if inlanding:
                    counttf[:] = []
                    cv2.rectangle(frame,(landingwidth[0],landingheight[0]),(landingwidth[1],landingheight[1]),(255,255,255),2)
                    countld.append(self.irises)
                    waittime = 3 - int(len(countld)/11)
                    if waittime > 0:
                        g.write("waiting\r\n")
                        cv2.putText(frame, 'landing in: ' + str(waittime) + 'sec', (landingwidth[0],int(landingheight[1]/2)), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    if waittime <= 0:
                        g.write("landing\r\n")
                        cv2.putText(frame, 'landing,please wait', (landingwidth[0],int(landingheight[1]/2)), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                       
                      
                if lost_track:
                    self.irises = self.classifier.get_irises_location(gray)
            else:  # cannot track for some reason -> find irises
                self.irises = self.classifier.get_irises_location(gray)
            show_image_with_data(frame, self.blinks, self.landblinks, self.irises, self.image_source.capture)
            k = cv2.waitKey(30) & 0xff
            old_gray = gray.copy()

        self.image_source.release()
        cv2.destroyAllWindows()


eyeris_detector = EyerisDetector(image_source=ImageSource(), classifier=CascadeClassifier(),
                                 tracker=LucasKanadeTracker())
eyeris_detector.run()

g.close()
