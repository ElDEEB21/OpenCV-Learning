import cv2 as cv
import numpy as np

blank = np.zeros((400, 400), dtype='uint8')

rectangle = cv.rectangle(blank.copy(), (30, 30), (370, 370), 255, -1)
circle = cv.circle(blank.copy(), (200, 200), 200, 255, -1)

cv.imshow('Rectangle', rectangle)
cv.imshow('Circle', circle)

## Bitwise And (Intersection)
bitwise_and = cv.bitwise_and(rectangle, circle)
cv.imshow('Bitwise And', bitwise_and)

## Bitwise Or (Union)
bitwise_or = cv.bitwise_or(rectangle, circle)
cv.imshow('Bitwise Or', bitwise_or)

## Bitwise Xor (Symmetric Difference -> non-intersecting areas)
bitwise_xor = cv.bitwise_xor(rectangle, circle)
cv.imshow('Bitwise Xor', bitwise_xor)

## Bitwise Not (Inversion)
bitwise_not_rectangle = cv.bitwise_not(rectangle)
bitwise_not_circle = cv.bitwise_not(circle)
cv.imshow('Bitwise Not Rectangle', bitwise_not_rectangle)
cv.imshow('Bitwise Not Circle', bitwise_not_circle)

cv.waitKey(0)

