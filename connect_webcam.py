# python connect_webcam.py

import cv2
print(cv2.__version__)
vidcap = cv2.VideoCapture(0)
success, image = vidcap.read()
count = 0
success = True
while success:
  # Capture frame-by-frame
  ret, frame = vidcap.read()

  # Our operations on the frame come here
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Display the resulting frame
  cv2.imshow('frame', gray)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    print(cv2.waitKey(1))
    break

  if count == 500:
    break

  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
  success, image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1

# When everything done, release the capture
vidcap.release()
cv2.destroyAllWindows()
