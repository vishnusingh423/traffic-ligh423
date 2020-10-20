import cv2 # computer vision library
import helpers # helper functions
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # for loading in images
get_ipython().run_line_magic('matplotlib', 'inline')
IMAGE_DIR_TRAINING = "traffic_light_images/training/"
IMAGE_DIR_TEST = "traffic_light_images/test/"
IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)
n = 0
selected_label = IMAGE_LIST[n][1]
while selected_label != "yellow":
    n += 1
    selected_label = IMAGE_LIST[n][1]

selected_image = IMAGE_LIST[n][0]
plt.imshow(selected_image)
print(selected_label)
print(n)
def standardize_input(image):
    standard_im = np.copy(image)    
    return cv2.resize(standard_im, (32,32))

def one_hot_encode(label):
    colors = {"red" : 0,
             "yellow" : 1,
             "green" : 2}
    one_hot_encoded = [0]*len(colors)
    one_hot_encoded[colors[label]] = 1
    
    return one_hot_encoded
import test_functions
tests = test_functions.Tests()
tests.test_one_hot(one_hot_encode)
def standardize(image_list):
    standard_list = []
    for item in image_list:
        image = item[0]
        label = item[1]
        standardized_im = standardize_input(image)
        one_hot_label = one_hot_encode(label)    
        standard_list.append((standardized_im, one_hot_label))
        
    return standard_list
STANDARDIZED_LIST = standardize(IMAGE_LIST)
n = 800
selected_image = STANDARDIZED_LIST[n][0]
plt.imshow(selected_image)
selected_label = STANDARDIZED_LIST[n][1]
print(selected_label)
image_num = 0
test_im = STANDARDIZED_LIST[image_num][0]
test_label = STANDARDIZED_LIST[image_num][1]
hsv = cv2.cvtColor(test_im, cv2.COLOR_RGB2HSV)

print('Label [red, yellow, green]: ' + str(test_label))
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
ax1.set_title('Standardized image')
ax1.imshow(test_im)
ax2.set_title('H channel')
ax2.imshow(h, cmap='gray')
ax3.set_title('S channel')
ax3.imshow(s, cmap='gray')
ax4.set_title('V channel')
ax4.imshow(v, cmap='gray')
def create_feature(rgb_image):
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    kernel = np.array([[ -4, -4, -4], 
                       [ -4, 32, -4], 
                       [ -4, -4, -4]])
    s_edges = cv2.filter2D(s, -1, kernel)
    blur = np.array([[ 1/9, 1/9, 1/9], 
                       [ 1/9, 1/9, 1/9], 
                       [ 1/9, 1/9, 1/9]])
    s_blur = cv2.filter2D(s, -1, kernel)
    for i in range(20):
        s_blur = cv2.filter2D(s_blur, -1, blur)
    
    #Create mask based on blurred edges in s
    s_blur_avg = int(np.sum(s_blur)/(len(s_blur)*len(s_blur[0])))
    s_blur_std = int(np.std(s_blur))
    s_mask = np.greater(s_blur, s_blur_avg+s_blur_std)
    
    #apply the mask to v
    v_mask = v
    v_mask[s_mask == 0] = [0]
    v_top = np.sum(v_mask[0:15])
    v_middle = np.sum(v_mask[7:23])
    v_bottom = np.sum(v_mask[15:31])
    v_sum = v_top + v_middle + v_bottom
    feature = [v_top/v_sum, v_middle/v_sum, v_bottom/v_sum]
    
    return feature

image_num = 723
test_im = STANDARDIZED_LIST[image_num][0]
create_feature(test_im)
def create_feature2(rgb_image):

    ##Convert image to HSV color space
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    
    # Detect edges in S     
    # 3x3 edge detection filters
    kernel = np.array([[ -4, -4, -4], 
                       [ -4, 32, -4], 
                       [ -4, -4, -4]])
    s_edges = cv2.filter2D(s, -1, kernel)
    
    # Blur edges.  Need to blur enough so that areas with signification changes in saturation bleed into each other
    blur = np.array([[ 1/9, 1/9, 1/9], 
                       [ 1/9, 1/9, 1/9], 
                       [ 1/9, 1/9, 1/9]])
    s_blur = cv2.filter2D(s, -1, kernel)
    for i in range(20):
        s_blur = cv2.filter2D(s_blur, -1, blur)
    
    #Create mask based on blurred edges in s
    s_blur_avg = int(np.sum(s_blur)/(len(s_blur)*len(s_blur[0])))
    s_blur_std = int(np.std(s_blur))
    s_mask = np.greater(s_blur, s_blur_avg+s_blur_std)
    
    #apply the mask to h
    h_mask = h
    h_mask[s_mask == 0] = [0]
    
    feature = np.sum(h_mask/360)/np.sum(s_mask)
    
    return feature

image_num = 2
test_im = STANDARDIZED_LIST[image_num][0]
create_feature2(test_im)
def estimate_label(rgb_image):
    
    feature = np.array(create_feature(rgb_image))
    predicted_label = [0, 0, 0]
    if create_feature2(rgb_image) > 0.38:
        predicted_label[0] = 1
    else:
        predicted_label[feature.argmax(axis=0)] = 1
    
    return predicted_label   

image_num = 723
test_im = STANDARDIZED_LIST[image_num][0]
estimate_label(test_im)
TEST_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TEST)

# Standardize the test data
STANDARDIZED_TEST_LIST = standardize(TEST_IMAGE_LIST)

# Shuffle the standardized test data
random.shuffle(STANDARDIZED_TEST_LIST)


# In[60]:


# Constructs a list of misclassified images given a list of test images and their labels
# This will throw an AssertionError if labels are not standardized (one-hot encoded)

def get_misclassified_images(test_images):
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in test_images:

        # Get true data
        im = image[0]
        true_label = image[1]
        assert(len(true_label) == 3), "The true_label is not the expected length (3)."

        # Get predicted label from your classifier
        predicted_label = estimate_label(im)
        assert(len(predicted_label) == 3), "The predicted_label is not the expected length (3)."

        # Compare true and predicted labels 
        if(predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))
            
    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels


# Find all misclassified images in a given test set
MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)

# Accuracy calculations
total = len(STANDARDIZED_TEST_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total

print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))
n = 6
selected_image = MISCLASSIFIED[n][0]
print(create_feature2(selected_image))
plt.imshow(selected_image)
import test_functions
tests = test_functions.Tests()

if(len(MISCLASSIFIED) > 0):
    # Test code for one_hot_encode function
    tests.test_red_as_green(MISCLASSIFIED)
else:
    print("MISCLASSIFIED may not have been populated with images.")


# In[ ]:




