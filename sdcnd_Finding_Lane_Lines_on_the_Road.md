# Finding Lane Lines on the Road

## Objective

Author : Mouad HADJI, Oct 11th, 2017

In this project, I will explain how to identify lane lines on the road. I will use primarily Python and OpenCV. The end goal will be to showcase how the code can find lane lines "live" in a video.

Several techniques are used :

* Color Selection
* Gaussian smoothing
* Canny Edge Detection
* Region of Interest Selection
* Hough Transform Line Detection
* etc.
	
## Color Manipulation

To start with, images came as screenshots from an onboard video feed.

![lane1](/test_images/solidWhiteRight.jpg) 
![lane2](/test_images/solidYellowLeft.jpg)

The first step that I took was to turn the image to grayscale to make it easier to work, namely to reduce the number of channels to work with. However, when dealing with more challenging images such as lane lines that are on non-contrasting backgrounds (white or gray tarmac), the eventual pipeline for lane linea detection does not perform well. In order to improve the performance, I switched to using [hue, saturation, and light](https://en.wikipedia.org/wiki/HSL_and_HSV#HSL) color space, which is better able to highlight the yellow and white lane lines.

_Grayscale_
![gray](/step_images/grayscale.png) 

_HSL color space_
![hsl](/step_images/hls.png)

 In the above image, we can see that the yellow lane is very clearly highlighted and the white line markings are also captured well when compared to the grayscale image. However, to further improve the performance of the processing pipeline, we can also select out the colors that we know we care about (in this case the yellow and white lines, which are now blue and green)

```Python
## color selection for yellow and white, using the HSL color space
def select_white_and_yellow(image):
    # converting to HLS
    hls_image = rgb_to_hls(image)

    # defining the bounds of each color
    lower_white = np.uint8([20,200,0])
    upper_white = np.uint8([255,255,255])
    lower_yellow = np.uint8([10,50,100])
    upper_yellow = np.uint8([100,255,255])
    
    # extracting masks
    white_mask = cv2.inRange(hls_image, lower_white, upper_white)
    yellow_mask = cv2.inRange(hls_image, lower_yellow, upper_yellow)
    
    # combining masks
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

    return cv2.bitwise_and(image, image, mask = combined_mask)
```
In the above code, I first convert the color map from RGB to HSL. Then I use the `inRange` function provided by OpenCV to select colors that fall into the white and yellow ranges. After that I combine the white and yellow masks together with the `bitwise_or` function. 

With the above HSL image, we can now try to isolate the yellow and the white lines. While there are many different techniques that can be utilized here, I chose to detect the edges within the image using the Canny edge detection algorithm. 

## Edge Detection

_HSL color selection_
![hsl](/step_images/challengeShadow_hsl.jpg)

Given the above image, the goal is to pick out the lane lines. In order to do this, I use the [canny edge detector](https://en.wikipedia.org/wiki/Canny_edge_detector) algorithm. In short, the algorithm:

	1. Applies a gaussian filter to the image to reduce noise
	2. Finds the gradients in both the horizontal and vertical directions
	3. Non-maximum supression, which is a way to thin the detected edges by only keeping the maximum gradient values and setting others to 0
	4. Determining potential edges by checking against a threshold 
	5. Finish cleaning potential edges by checking in the potential edge is connected to an actual edge
	
While, the canny edge detector automatically applies [gaussian blur](https://en.wikipedia.org/wiki/Gaussian_blur), I applied gaussian blur outside of the edge detector so that I could have more freedom with the kernel parameter. After running the image through the blurring and edge detection functions, the image is as follows. Note, the input image to this is the HSL color converted image. 

_HSL color selection with canny edge detection_
![hsl_canny](/step_images/canny.png.png)

With the image above, we see that the lane lines are pretty well identified. It took a bit of trial and error to find suitable thresholds for the canny edge detector though the creator John Canny recommended a ratio of 1:2 or 1:3 for the low vs. high threshold. Although the image above seems to mark the lane lines quite well, there is still a lot of noise surrounding the lane that we do not care about. In order to address this, we can apply a region mask to just keep the area that we know contains the lane lines. 

After applying the mask to the canny image, we get the following output : 

_HSL color selection with canny edge detection and region masking_
![region_canny](/step_images/challengeShadow_regioncanny.jpg) 
	
As shown above, the HSL version provides a very indication of the lane lines. Below are the functions used in processing the images.
	
```Python
def canny(img, low_threshold=50, high_threshold=150):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size=15):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
```

## Hough Line Transform

Now that we have a collection of edges, we need to identify the lane lines within the image. The [hough line transform](https://en.wikipedia.org/wiki/Hough_transform), which was first invented to identify lines within images, is great for this task. To learn more about this algorithm, this [blog](http://alyssaq.github.io/2014/understanding-hough-transform/) is a great resource. 

_HSL color selection with canny edge detection, region masking, and hough transform_
![Hough transform in action](/step_images/hough.png) 

The lane lines have now been highlighted and boxed with the red lines. There are quite a few parameters that needed to be adjusted, but after adjusting the parameters, the algorithm is able to pick out the lines quite well.


```Python
def hough_lines(img, rho=1, theta=np.pi/180, threshold=25, min_line_len=20, max_line_gap=300):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines
```

## Lane Line Averaging

Given the above image and specifically the hough lines, the goal now is to average of the detected lines into 2 very recognizable lines that represent the lane lines. For this purpose, we should come up with an averaged line for that.

Also, because some of the lane lines are only partially recognized, we will need to extrapolate the line to cover full lane line length.

In a nutshell, we need to display two lane lines: one for the left and the other for the right. The left lane should have a positive slope, and the right lane should have a negative slope. Therefore, we'll collect positive slope lines and negative slope lines separately and take averages without forgetting that the y coordinate is reversed in our pictures. Hence, Left image = negative slope and Right image = positive slope.

```Python
def average_lines(lines):
    left_lines    = []
    left_weights  = []
    right_lines   = []
    right_weights = []
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2==x1:
                continue # ignore a vertical line
            slope = (y2-y1)/(x2-x1)
            intercept = y1 - slope*x1
            length = np.sqrt((y2-y1)**2+(x2-x1)**2)
            if slope < 0 and length > 10:
                left_lines.append((slope, intercept))
                left_weights.append((length))
            elif slope > 0 and length > 10:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    
    left_lane  = np.dot(left_weights,  left_lines) /np.sum(left_weights)  if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights)> 0 else None
    
    return left_lane, right_lane # (slope, intercept), (slope, intercept)
```

The above function calculates the slope, intercept, and line length of each line segment. Now that we have the attributes of our lanes, there are 3 steps we need to go through in order to display the generated lines on our images :

- i. Find the end points of our lines so we can use the cv2.line function to generate a line (ii)
- ii. Generate the line points given the pixels points we calculated
- iii. Draw the lines on the images to make them clearly visible

Here are the results :

_Final processed image_
![hsl_final](/step_images/lanes.png)

It seems to have performed very well! 

## Applying Lane Finding to Videos

Now that we can identify and mark the lane lines within the image supplied, we can use the algorithm on a video, which is just a sequence of images. If we just apply the pipeline directly to the video, we get lane line highlights that are jittering and jumping across back and forth around the actual location of the lane line. While, the algorithm basically accomplishes the problem that we first set out to solve, we can definitely improve it.

Specifically, the lane lines coming from a video feed usually do not change dramatically from second to second. If we take this into account, we can "smooth" the lane lines plotted out by keeping a queue. With each frame of the video, we can pop off the oldest set of lane line endpoints. Then for all the remaining lane lines and the newest lane line, we take an average to get the "smoothed" lane line. 

Below is the code for the lane line detector :

The videos show that the new detector with the lane line averaging works quite nicely! Although if there are drastic changes the algorithm does not follow those changes until a bit later, we can fiddle with this by changing the size of the queue.

```Python
class lane_detector:
    def __init__(self):
        self.left_lines_historical  = []
        self.right_lines_historical = []
        
    def average_with_previous(self, line, lines):
        if line is not None:
            lines.append(line)
            
        if len(lines) > 0:
            line = np.mean(lines, axis=0, dtype=np.int32)
            line = tuple(map(tuple, line))
            
        return line    
        
        
    def pipeline(self, image):
        imshape = image.shape
        gray_image = grayscale(image)
        white_yellow_image = select_white_and_yellow(image)
        blur_gray = gaussian_blur(white_yellow_image)
        edge_image = canny(blur_gray)
        chosen_vertices = np.array([[(imshape[1]*0.05, imshape[0]), (imshape[1]*0.48, imshape[0]*0.6),
            (imshape[1]*0.52, imshape[0]*0.6), (imshape[1]*0.95, imshape[0])]], dtype=np.int32)
        masked_image = region_of_interest(edge_image, chosen_vertices)
        lines = hough_lines(masked_image)
        left_line, right_line = generate_lane_lines(image, lines)
        
        new_left_line  = self.average_with_previous(left_line,  self.left_lines_historical)
        new_right_line = self.average_with_previous(right_line, self.right_lines_historical)
        
        return draw_lane_lines(image, (new_left_line, new_right_line))
```

## Shortcomings & Next Steps

While the detector works fairly well for straight roads, there are limitations:

* Curved Roads
* Lane markings that are not yellow or white
* New lanes that could be on the ground and taken for lanes by mistake
	
In order to deal with these shortcomings, we would need to make the algorithm more robust to differences and maybe use other techniques to validate the identification of lanes such as deep learning for Computer Vision (CNNs for instance). We could also keep a dictionary of the most probable slopes, etc.