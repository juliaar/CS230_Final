# python connect_webcam.py

import cv2

def getFrame(count):
  # vidcap.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
  hasFrames, image = vidcap.read()
  if hasFrames:
    cv2.imwrite("frame" + str(count) + ".jpg", image)  # save frame as JPG file
  return hasFrames


vidcap = cv2.VideoCapture(0)
# automatically captures 30 fps (we want to show those but only save 3 fps)
#print(vidcap.get(cv2.CAP_PROP_FPS))
totcount = 0
count = 0

while(True):
    # Capture frame-by-frame
    ret, frame = vidcap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame', gray)

    if (totcount % 10) == 0:
      count = count + 1
      # Only want to store 5 images
      if count == 6:
        count = 1
      # Save frame
      getFrame(count)
      print(count)

    if totcount == 200:
      break

    totcount = totcount + 1

    # Stop if q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
vidcap.release()
cv2.destroyAllWindows()

