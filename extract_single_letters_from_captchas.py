import os
import os.path
import cv2
import glob
import imutils
import PIL


CAPTCHA_IMAGE_FOLDER = "captcha_images"
OUTPUT_FOLDER = "extracted_letter_images"


chop = 2

# Get a list of all the captcha images we need to process
captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
counts = {}

# loop over the image paths
for (i, captcha_image_file) in enumerate(captcha_image_files):
    print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))

    # grab the base filename as the text
    filename = os.path.basename(captcha_image_file)
    captcha_correct_text = os.path.splitext(filename)[0]
    print(filename)
    image = PIL.Image.open("captcha_images/"+filename).convert('1')
    width, height = image.size
    data = image.load()

    
    for y in range(height):
        for x in range(width):
    
            if data[x, y] > 128:
                continue
    
            
            total = 0
            for c in range(x, width):
    

                if data[c, y] < 128:
                    total += 1
    
                else:
                    break

            if total <= chop:
                for c in range(total):
                    data[x + c, y] = 255
    
            x += total
    
    

    for x in range(width):
        for y in range(height):
    

            if data[x, y] > 128:
                continue
    

            total = 0
    
            for c in range(y, height):
    
                if data[x, c] < 128:
                    total += 1
    
                else:
                    break
    
            if total <= chop:
                for c in range(total):
                    data[x, y + c] = 255
    
            y += total

    letter_image_regions=[]
    letter_image_regions.append((5, 1, 35, 50))        
    letter_image_regions.append((25, 1, 55, 50))        
    letter_image_regions.append((45, 1, 75, 50))        
    letter_image_regions.append((65, 1, 100, 50))        
    letter_image_regions.append((95, 1, 125, 50))        
    # Save out each letter as a single image
    for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box


        letter_image=image.crop((x,y,w,h))
        # Get the folder to save the image in
        save_path = os.path.join(OUTPUT_FOLDER, letter_text)

        # if the output directory does not exist, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # write the letter image to a file
        count = counts.get(letter_text, 1)
        p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        letter_image.save(p,"png")

        # increment the count for the current key
        counts[letter_text] = count + 1
