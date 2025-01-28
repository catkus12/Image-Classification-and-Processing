from PIL import Image, ImageFilter
from PIL import Image, ImageEnhance, ImageOps, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

def apply_blur_filter(image_path):
    try:
        img = Image.open(image_path)
        img_blurred = img.filter(ImageFilter.GaussianBlur(radius=10))

        img_blurred.save("blurred_image.jpg")
        print("Blurred image saved as 'blurred_image.jpg'.")

    except Exception as e:
        print(f"Error applying blur filter: {e}")

def apply_sharpen_filter(image_path):
    try:
        img = Image.open(image_path)
        img_sharpened = img.filter(ImageFilter.SHARPEN)

        img_sharpened.save("sharpened_image.jpg")
        print("Sharpened image saved as 'sharpened_image.jpg'.")

    except Exception as e:
        print(f"Error applying sharpen filter: {e}")

def apply_edge_detection_filter(image_path):
    try:
        img = Image.open(image_path)
        img_edges = img.filter(ImageFilter.FIND_EDGES)

        img_edges.save("edges_image.jpg")
        print("Edge-detected image saved as 'edges_image.jpg'.")

    except Exception as e:
        print(f"Error applying edge detection filter: {e}")

def apply_contour_filter(image_path):
    try:
        img = Image.open(image_path)
        img_contour = img.filter(ImageFilter.CONTOUR)

        img_contour.save("contour_image.jpg")
        print("Contour image saved as 'contour_image.jpg'.")

    except Exception as e:
        print(f"Error applying contour filter: {e}")

def valentines_filter(image_path):
    try:
        # Step 1: Open the image
        img = Image.open(image_path)
        img = img.convert("RGBA")  # Ensure transparency support

        # Step 2: Add a pink tint to the image
        pink_overlay = Image.new("RGBA", img.size, (255, 182, 193, 100))  # Light pink overlay with transparency
        img_with_tint = Image.alpha_composite(img, pink_overlay)  # Blend the pink tint onto the image

        # Step 3: Draw red and pink hearts
        draw = ImageDraw.Draw(img_with_tint)

        # Function to draw a heart shape
        def draw_heart_big(draw, x, y, size, color):
            # Define points for the heart shape
            top_left = (x - size, y - 60)
            top_right = (x + size, y - 60)
            bottom = (x, y + size * 1)
            draw.polygon([top_left, bottom, top_right], fill=color)  # Draw the bottom triangle
            draw.ellipse([x - size, y - size, x, y], fill=color)  # Left circle
            draw.ellipse([x, y - size, x + size, y], fill=color)  # Right circle

        def draw_heart_small(draw, x, y, size, color):
            # Define points for the heart shape
            top_left = (x - size, y - 20)
            top_right = (x + size, y - 20)
            bottom = (x, y + size * 1)
            draw.polygon([top_left, bottom, top_right], fill=color)  # Draw the bottom triangle
            draw.ellipse([x - size, y - size, x, y], fill=color)  # Left circle
            draw.ellipse([x, y - size, x + size, y], fill=color)  # Right circle
       
        # Draw hearts at different positions and sizes
        draw_heart_big(draw, 180, 200, 150, "red")  
        draw_heart_big(draw, 1200, 160, 150, "pink")  
        draw_heart_big(draw, 2800, 360, 150, "red")   

        draw_heart_small(draw, 100, 540, 50, "pink")  
        draw_heart_small(draw, 1700, 270, 50, "red")  
        draw_heart_small(draw, 2600, 400, 40, "pink")   

        # Step 4: Save the resulting image
        img_with_tint = img_with_tint.convert("RGB")
        img_with_tint.save("valentines_filter.jpg")
        print("Image with valentines filter saved as 'valentines_filter.jpg'.")

    except Exception as e:
        print(f"Error adding hearts and tint: {e}")

if __name__ == "__main__":
    image_path = "Gordon.jpg"  # Replace with the path to your image file

    # Apply each filter and save as separate images
   # apply_blur_filter(image_path)           # Blurred image
   # apply_sharpen_filter(image_path)        # Sharpened image
   # apply_edge_detection_filter(image_path) # Edge-detected image
   # apply_contour_filter(image_path)        # Contour image
    valentines_filter(image_path)