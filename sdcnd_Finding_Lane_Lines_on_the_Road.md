# Finding Lane Lines on the Road

## Introduction

Author : Mouad HADJI, Oct 11th, 2017

In this project, I will explain how to identify lane lines on the road. I will use primarily Python and OpenCV. The end goal will be to showcase how the code can find lane lines "live" in a video (which is pretty cool).

Several techniques are used :

* Color Selection
* Gaussian smoothing
* Canny Edge Detection
* Region of Interest Selection
* Hough Transform Line Detection
* etc.

We will detail the basic principles in this summary.
	
## Color Selection

The test images have the following format. They are quite simple and except the fact that the lanes might be yellow, the luminosity is very good in each of them.

![lane1](/test_images/solidWhiteRight.jpg) 

The first step I took was to grayscale the image to make it easier to work, namely to reduce the number of channels to work with. However, when dealing with more challenging images such as lane lines that are on non-contrasting backgrounds (white or gray tarmac), the lane detection pipeline for lane linea detection struggles. In order to improve the performance, I switched to using [hue, saturation, and light](https://en.wikipedia.org/wiki/HSL_and_HSV#HSL) color space, which is better able to highlight the yellow and white lane lines.

_Grayscale_
![gray](/step_images/grayscale.png) 

_HSL color space_
![hsl](/step_images/hls.png)

 In the above image, white and yellow lanes are much better captured compared to the grayscale image. However, to further improve the performance of the processing pipeline, we can also select out the colors that we know we care about. The assumption here is that we will not encounter any lane of "different" colors which is not obvious.

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

In the above code, the first command converts color encoding from RGB to HSL. Then I use the `inRange` function provided by OpenCV to select colors that fall into the white and yellow ranges. After that I combine the white and yellow masks together with the `bitwise_or` function. The result is very helpful for the rest of the pipeline.

_HSL color selection_
![hsl](/step_images/yellow_and_white.png)


## Edge Detection

Given the above image, the goal is to pick out the lane lines. In order to do this, I use the [canny edge detector](https://en.wikipedia.org/wiki/Canny_edge_detector) algorithm. In short, the algorithm:

	1. Applies a gaussian filter to the image to reduce noise
	2. Finds the gradients in both the horizontal and vertical directions
	3. Non-maximum supression, which is a way to thin the detected edges by only keeping the maximum gradient values and setting others to 0
	4. Determining potential edges by checking against a threshold 
	5. Finish cleaning potential edges by checking in the potential edge is connected to an actual edge
	
While, the canny edge detector automatically applies [gaussian blur](https://en.wikipedia.org/wiki/Gaussian_blur), I applied gaussian blur outside of the edge detector to add a degree of freedom in the customization of the pipeline.

_HSL color selection with canny edge detection_
![hsl_canny](/step_images/canny.png)

With the image above, we see that the lane lines are pretty well identified. After applying the mask to the canny image, we get the following output : 

_HSL color selection with canny edge detection and region masking_
![region_canny](/step_images/canny_masked.png) 
	
As shown above, the HSL version provides a very indication of the lane lines. Below are the functions used in processing the images.
	
```Python
def gaussian_blur(img, kernel_size=15):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img, low_threshold=50, high_threshold=150):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

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

After collecting the edges, the next task is to identify the lane lines. The [hough line transform](https://en.wikipedia.org/wiki/Hough_transform), which was first invented to identify lines within images, is very helpful for this matter.

_HSL color selection with canny edge detection, region masking, and hough transform_
![Hough transform in action](/step_images/hough.png) 

The lane lines are now detected !


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

Eventually, given the above image and specifically the hough lines, the goal now is to average of the detected lines into 2 very recognizable lines that represent the lane lines. For this purpose, we should come up with an averaged line for that.

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

Here are the final results after going through the pipeline :

_Final processed image_
![hsl_final](/step_images/lanes.png)

It seems to have performed very well! 

## Upgrading to Videos

A video is just a sequence of images, hence we are totally capable of using our lane detection algorithm to videos ! The first trials we quite successful but I realized rapidly that I will have to deal with a lot of jittering and unstability. The best practice here is to keep averaging the lanes with historical data which works very welll when the road is not curved. We can probably keep the n last lanes to play with this paramter. This is totally an improvement that I should look at bringing in the next steps.

Below is the code for the lane line detection pipeline :

```Python
class lane_detection:
    
    def __init__(self):
        self.left_lines_historical  = []
        self.right_lines_historical = []
    
    # this has been added after noticing jitter on the videos
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

While the detection algorithm works well for straight roads like we mentioned before, the roadblocks are the followings :

* Curved Roads
* Lane marked with other colors
* New lines detected on the ground that could be taken for lanes by mistake and screwing the average.
	
In order to deal with these shortcomings, we would need to make the algorithm more robust to differences and maybe use other techniques to validate the identification of lanes such as deep learning for Computer Vision (Convolutional Neural Netowrk is a famous technique). We could also keep a dictionary of the most probable slopes, etc.