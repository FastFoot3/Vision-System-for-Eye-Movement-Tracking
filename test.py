import cv2

hewwow = 'Hewwow Wowd :3'
print(hewwow)

img = cv2.imread('logo.png', -1)

cv2.imshow(':3', img)
cv2.waitKey(0)
cv2.destroyAllWindows()