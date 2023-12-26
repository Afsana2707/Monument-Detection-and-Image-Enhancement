import cv2
import numpy as np
import base64
import math


final_image = ''
def convert_to_grayscale(image_path):
    image = cv2.imread(image_path)
    if image is not None:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        retval, buffer = cv2.imencode('.jpg', gray_image)
        if retval:
            base64_image = base64.b64encode(buffer).decode()
            return base64_image
    return None


def edge_detection(image_path, detection_type):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is not None:
        edges = None
        if detection_type == "canny":
            edges = cv2.Canny(image, 100, 200) 
        elif detection_type == "sobel":
            sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = cv2.magnitude(sobel_x, sobel_y)
            edges = cv2.convertScaleAbs(magnitude)
        elif detection_type == "scharr":
            scharr_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
            scharr_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)
            magnitude = cv2.magnitude(scharr_x, scharr_y)
            edges = cv2.convertScaleAbs(magnitude)
            
        if edges is not None:
            retval, buffer = cv2.imencode('.jpg', edges)
            if retval:
                base64_image = base64.b64encode(buffer).decode()
                return base64_image

    return None  

def rotation(image, angle):
    if image != None:
        if angle is None:
            angle = 90
        image = cv2.imread(image, cv2.COLOR_BGR2RGB)
        rows, cols, _ = image.shape
        center = (cols // 2, rows // 2)
        rotated_image = np.zeros_like(image)
        for i in range(rows):
            for j in range(cols):
                x = int((i - center[1]) * np.cos(np.radians(angle)) - (j - center[0]) * np.sin(np.radians(angle)) + center[1])
                y = int((i - center[1]) * np.sin(np.radians(angle)) + (j - center[0]) * np.cos(np.radians(angle)) + center[0])
                if 0 <= x < rows and 0 <= y < cols:
                    rotated_image[i, j] = image[x, y]
        retval, buffer = cv2.imencode('.jpg', rotated_image)
        if retval:
            base64_image = base64.b64encode(buffer).decode()
            return base64_image
    return None



def transformation_func(image, transform="translate", tx=0, ty=0, c=0, d=0, e=1, f=0):
    if image != None:
        image = cv2.imread(image, cv2.COLOR_BGR2RGB)
        rows, cols, _ = image.shape
        transformation = np.zeros_like(image)
        if transform == "translate":
            if tx is None:
                tx = 20
            if ty is None:
                ty  = 20
            transformation = np.zeros_like(image)
            tx = int(tx)
            ty = int(ty)
            for i in range(rows):
                for j in range(cols):
                    if 0 <= i + tx < rows and 0 <= j + ty < cols:
                        transformation[i + tx, j + ty] = image[i, j]
        elif transform == "affine":
            tranformation = np.zeros_like(image)
            ty = 0.5
            for i in range(rows):
                for j in range(cols):
                    x = int(tx * i + ty * j + c)
                    y = int(d * i + e * j + f)

                    if 0 <= x < rows and 0 <= y < cols:
                        transformation[i, j] = image[x, y]
        elif transform == "shear":
            tranformation = np.zeros_like(image)
            tx = 0.2
            ty = 0.1
            for i in range(rows):
                for j in range(cols):
                    x = int(i + ty * j)
                    y = int(tx * i + j)

                    if 0 <= x < rows and 0 <= y < cols:
                        transformation[i, j] = image[x, y]
        elif transform == "deformation":
            random_displacement = np.random.normal(0, ty, size=(rows, cols, 2))
            grid_x, grid_y = np.meshgrid(np.arange(cols), np.arange(rows))
            distorted_x = grid_x + tx * random_displacement[:, :, 0]
            distorted_y = grid_y + tx * random_displacement[:, :, 1]

            transformation = np.zeros_like(image)

            for i in range(rows):
                for j in range(cols):
                    x = int(distorted_x[i, j])
                    y = int(distorted_y[i, j])

                    if 0 <= x < rows and 0 <= y < cols:
                        transformation[i, j] = image[x, y]
        retval, buffer = cv2.imencode('.jpg', transformation)
        if retval:
            base64_image = base64.b64encode(buffer).decode()
            return base64_image
    return None


def median_blur(image, blur_value):
    print("image is none")
    if image is not None:
        image = cv2.imread(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred_image = cv2.medianBlur(gray_image, blur_value)
        print("median blurr")            
        retval, buffer = cv2.imencode('.jpg', blurred_image)
        if retval:
            base64_image = base64.b64encode(buffer).decode()
            return base64_image

    return None


def bilateral_filter(image, d, sigma_color, sigma_space):
    if image is not None:
        image = cv2.imread(image, cv2.COLOR_BGR2RGB)
        # Convert the color image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply bilateral filter to the grayscale image
        filtered_image = cv2.bilateralFilter(gray_image, d, sigma_color, sigma_space)

        retval, buffer = cv2.imencode('.jpg', filtered_image)
        if retval:
            base64_image = base64.b64encode(buffer).decode()
            return base64_image

    return None


def save(image):
    final_image = image
    return final_image