
import os
import numpy as np
from PIL import Image
import base64
import io
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_screen():
    """Create a test Game Boy screen with a pattern to verify display quality"""
    width, height = 160, 144

    # Create a base image with GB green color
    img = Image.new('RGB', (width, height), color=(155, 188, 15))

    # Create drawing context
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)

    # Draw a grid pattern
    for x in range(0, width, 8):
        for y in range(0, height, 8):
            if (x + y) % 16 == 0:
                draw.rectangle([x, y, x+7, y+7], fill=(15, 56, 15))

    # Draw some text
    from PIL import ImageFont
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()

    draw.text((10, 10), "GAME BOY", fill=(0, 0, 0), font=font)
    draw.text((10, 30), "Display Test", fill=(0, 0, 0), font=font)

    # Draw Nintendo logo-like shape
    draw.rectangle([60, 60, 100, 80], fill=(0, 0, 0))

    # Save the image
    img.save('gameboy_test_screen.png')
    logger.info("Saved test screen to gameboy_test_screen.png")

    # Convert to base64 for web display
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Save the base64 string to a file for testing
    with open('test_screen_base64.txt', 'w') as f:
        f.write(img_base64)

    logger.info("Test complete. You can view the image in gameboy_test_screen.png")
    logger.info("Base64 version saved to test_screen_base64.txt for web testing")

    return img_base64

if __name__ == "__main__":
    print("Creating test Game Boy screen...")
    create_test_screen()
    print("Done! Check the output files.")
