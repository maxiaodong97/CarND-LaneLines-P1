# **Finding Lane Lines on the Road** 

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

High level, I use 8 steps pipeline to find lane lines on the road. 

 0. Load the image from file. 
 1. Use color selection to find only white and yellow color
 2. Convert th gray image for canny edge detection
 3. Filter out high frequency noises using gaussian blur. 
 4. Detect edge using canny algorithm.
 5. Selection region where lane lines may possible sit.
 6. Find out the lines using hough algorithm, find the complete extended right and left line
 7. Draw the line and original image together.

Here is the output for each step for all the test_images
![alt text](https://github.com/maxiaodong97/CarND-LaneLines-P1/blob/master/test_images_out/total.png "Pipeline Summary")

Here is the explaination of each step.

##### 0. First I use  matplotlib.image.imread to read image file to python image object.
##### 1. I filter the lane based on color. I found this can reduce some amount of time in later steps. The filter is based on opencv helper function 
```python
     mask = cv2.inRange(image, np.array([200, 200, 0]), np.array([255, 255, 255]))
    return cv2.bitwise_and(image, image, mask = mask)
```
I use low threshold [200, 200, 0] and high threshold [255, 255, 255] which covers both yellow and white space. 
##### 2. I convert the image to gray image, using helper function grayscale(). 
##### 3. I apply the gaussian_blur() helper function, using kernal size 13. In the test, I found this doesn't change result a lot.
##### 4. I apply canny edge detection using low threshold 50 and high threshold 150 suggested in the class.
##### 5. I select region that lane lane usually appeared. In real scenario, camera usually mounted in a fixed position and road is flat. So this is a valid assumption that lane lane appears only in certain area. In my case, I picked a trapezium region as: 
```python
    rows, cols = image.shape[:2]
    bottom_left  = [cols*0.1, rows*0.9]
    bottom_right = [cols*0.9, rows*0.9]
    top_left     = [cols*0.4, rows*0.6]
    top_right    = [cols*0.6, rows*0.6]
```
##### 6.I call hough_lines with tuned parameters: rto 1, theta=np.pi/180, threshold=20, min_line_len=20, max_line_gap=200. 
In order to draw a single line on the left and right lanes, I modified the draw_lines() function by approximating the left and right lane as two straight line. First I split the lines returned by hough_lines() function into two sets, left and right lines based on the slope.  For each set, I use fitLine() find the best approximation of line points. The result returned by fitLine() is vector and a point. Which can be used to calculate the line slope and intercept. Once I get the left line and right line, I will draw two lines from y1 to y2. Which is the lowest and highest possible height in the images. Again I make assumption that camera is fixed and road is flat. Here is the example code: 

```python
def split_lane(img, lines):
    # slope threshold
    SLOPE_THRESHOLD = 0.1 # use this to filter the horizontal line. (show as noise in solidYellowLeft.mp4)
    left_points = []
    right_points = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2==x1:
                continue # ignore the vertical lines
            slope = (y2-y1)/(x2-x1)
            if slope < -SLOPE_THRESHOLD:
                left_points.append([x1, y1])
                left_points.append([x2, y2])
            elif slope > SLOPE_THRESHOLD:
                right_points.append([x1, y1])
                right_points.append([x2, y2])
    left_cnt = np.array(left_points).reshape((-1,1,2)).astype(np.int32)
    right_cnt = np.array(right_points).reshape((-1,1,2)).astype(np.int32)
    # use fitLine to find the line that best cover all the line points.
    left_line = cv2.fitLine(left_cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    right_line = cv2.fitLine(right_cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    return left_line, right_line

def draw_extend_line(img, line, y1, y2, color, thickness):
    [vx,vy,x,y] = line
    if vx == 0:
        return
    m = vy / vx
    b = y - m * x
    x1 = (y1 - b) / m
    x2 = (y2 - b) / m
    if math.isnan(x1) or math.isnan(x2):
        return
    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    if lines is None or len(lines) == 0:
        return 

    left_line, right_line = split_lane(img, lines)

    # Extend to endpoints
    rows, cols = img.shape[:2]
    y1 = rows
    y2 = rows * 0.6
    draw_extend_line(img, left_line, y1, y2, color, thickness)
    draw_extend_line(img, right_line, y1, y2, color, thickness)
```
A few corner case to catch during my testing with videos.  One is line slope could be vertical or horizontal. In practical scenario, vertical line could be possible but rare. Horizontal line is most likely a false line. So I filter them out by slope threshold.  I also found x1 or x2 could be NaN after calcuation in some video frame, so I skipped them.

##### 7. Finally I combined lines and original image together. 

### 2. Identify potential shortcomings with your current pipeline

A few shortcomings in my solution: 
1. I use straight lines to approximate the left lane and right lane. This is most likely not true, as road may be curved. 
2. I define the region based on assumption that the camera is fixed and the road is flat. 
3. More parameter tunning is required to find out the best for the most scenarios.
4. If the slope changed dramatically in a video frame, this is most likely to be false identified. We should probably filter it out. 

### 3. Suggest possible improvements to your pipeline

A possible improvement for above shortcomings would be: 

1. Use spline interpolation interp1d() to make curved line in stead of straight line. I have tried this approach, but it looks hit some boundry errors using scipy as a limitation. Probably I can use some other library.
2. Use color selection to identify the road area, eg: [0, 0, 0] to [100, 100, 100] to mark the road and find the road line first as the region. This probably will not work for wild road. Another solution is to build 3-D model from frames, so we get depth information and from there to derive the drivable region 
3. Need to extract some images from video as samples to find better parameters.
4. Smooth out the high frequency slope across the video frames. 
