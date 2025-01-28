import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt


model = MobileNetV2(weights="imagenet")

# Function to add a black box to the image
def add_black_box(image, x, y, width, height):
    occluded_image = image.copy()
    # Ensure the box dimensions do not exceed the image dimensions
    height = min(height, image.shape[0] - y)
    width = min(width, image.shape[1] - x)
    occluded_image[y:y+height, x:x+width] = 0  # Add a black box
    return occluded_image

# Function to blur a region in the image
def blur_region(image, x, y, width, height):
    occluded_image = image.copy()
    # Ensure the blur region does not exceed the image dimensions
    height = min(height, image.shape[0] - y)
    width = min(width, image.shape[1] - x)
    region = occluded_image[y:y+height, x:x+width]
    blurred_region = cv2.GaussianBlur(region, (75, 75), 0)
    occluded_image[y:y+height, x:x+width] = blurred_region
    return occluded_image

# Function to add random noise to a region in the image
def add_noise_box(image, x, y, width, height):
    occluded_image = image.copy()
    # Ensure the noise region does not exceed the image dimensions
    height = min(height, image.shape[0] - y)
    width = min(width, image.shape[1] - x)
    noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    occluded_image[y:y+height, x:x+width] = noise
    return occluded_image

def classify_image(image_path):
    """Classify an image and display the predictions."""
    try:
         img = image.load_img(image_path, target_size=(224, 224))
        
         img_array = image.img_to_array(img)
        
         img_array = preprocess_input(img_array)
         img_array = np.expand_dims(img_array, axis=0)

         predictions = model.predict(img_array)

         decoded_predictions = decode_predictions(predictions, top=3)[0]

         print("Top-3 Predictions:")
         for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
             print(f"{i + 1}: {label} ({score:.2f})")

    except Exception as e:
         print(f"Error processing image: {e}")

# def get_grad_cam(model, img_array, class_index):
#     """
#     Generates a Grad-CAM heatmap for the specified class index.
#     Args:
#         model: The trained model (e.g., MobileNetV2).
#         img_array: The preprocessed image array (shape: (1, 224, 224, 3)).
#         class_index: The index of the class to visualize.
#     Returns:
#         A heatmap (2D array) highlighting the important regions.
#     """
#     grad_model = tf.keras.models.Model(
#         [model.inputs], 
#         [model.get_layer("Conv_1").output, model.output]
#     )
    
#     # Get gradients of the predicted class with respect to the last convolutional layer
#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_array)
#         loss = predictions[:, class_index]

#     grads = tape.gradient(loss, conv_outputs)
    
#     # Compute the mean intensity of the gradients across the channels
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
#     # Multiply the gradients by the activation map (element-wise)
#     conv_outputs = conv_outputs[0]
#     heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)
    
#     # Normalize the heatmap to a range of 0 to 1
#     heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
#     return heatmap.numpy()

# def display_grad_cam(img_path, heatmap, alpha=0.4, save_path="grad_cam_result.jpg"):
#     """
#     Displays the Grad-CAM heatmap superimposed on the original image and saves it.
#     Args:
#         img_path: Path to the input image.
#         heatmap: The Grad-CAM heatmap.
#         alpha: Opacity of the heatmap overlay (default: 0.4).
#         save_path: Path to save the resulting image (default: 'grad_cam_result.jpg').
#     """
#     # Load the original image
#     img = image.load_img(img_path)
#     img = np.array(img)

#     # Resize heatmap to match the image size
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = tf.image.resize(heatmap[..., tf.newaxis], (img.shape[0], img.shape[1]))
#     heatmap = tf.squeeze(heatmap).numpy()

#     # Apply the heatmap on the image
#     heatmap_color = plt.cm.jet(heatmap)[:, :, :3]  # Use jet colormap
#     superimposed_img = heatmap_color * alpha + img / 255.0

#     # Save the Grad-CAM result
#     superimposed_img = np.uint8(superimposed_img * 255)  # Convert back to pixel values (0–255)
#     plt.imsave(save_path, superimposed_img)

#     # Plot the original image and Grad-CAM result
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.title("Original Image")
#     plt.imshow(img)
#     plt.axis("off")

#     plt.subplot(1, 2, 2)
#     plt.title("Grad-CAM")
#     plt.imshow(superimposed_img)
#     plt.axis("off")

#     plt.tight_layout()
#     plt.show()

# def classify_image(image_path):
#     """Classify an image and display the predictions with Grad-CAM."""
#     try:
#         # Load and preprocess the image
#         img = image.load_img(image_path, target_size=(224, 224))
#         img_array = image.img_to_array(img)
#         img_array = preprocess_input(img_array)
#         img_array = np.expand_dims(img_array, axis=0)

#         # Get predictions
#         predictions = model.predict(img_array)
#         decoded_predictions = decode_predictions(predictions, top=3)[0]

#         # Print the predictions
#         print("Top-3 Predictions:")
#         for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
#             print(f"{i + 1}: {label} ({score:.2f})")

#         # Generate Grad-CAM for the top prediction
#         class_index = np.argmax(predictions[0])  # Get the index of the top predicted class
#         heatmap = get_grad_cam(model, img_array, class_index)

#         # Display the Grad-CAM
#         display_grad_cam(image_path, heatmap)

#     except Exception as e:
#         print(f"Error processing image: {e}")

if __name__ == "__main__":
    image_path = "Gordon.jpg"  
    img = cv2.imread(image_path)  # Load the original image

    # Calculate image dimensions for medium-sized box
    box_width = 2000 
    box_height = 1000 
    x = 1500 
    y = 1000 

    # Apply black box occlusion
    occluded_img_black = add_black_box(img, x, y, box_width, box_height)
    cv2.imwrite('occluded_black_box.jpg', occluded_img_black)

    # Apply blur region
    occluded_img_blur = blur_region(img, x, y, box_width, box_height)
    cv2.imwrite('occluded_blur.jpg', occluded_img_blur)

    # Apply noise box
    occluded_img_noise = add_noise_box(img, x, y, box_width, box_height)
    cv2.imwrite('occluded_noise.jpg', occluded_img_noise)

    # Classify one of the occluded images
    print("Box:")
    classify_image('occluded_black_box.jpg')

    print("Blur:")
    classify_image('occluded_blur.jpg')
    
    print("Noise:")
    classify_image('occluded_noise.jpg')