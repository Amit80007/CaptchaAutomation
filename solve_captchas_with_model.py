from keras.models import load_model
from resize import resize_to_fit
from imutils import paths
import numpy as np
import imutils
import cv2
import pickle


MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"
CAPTCHA_IMAGE_FOLDER = "captcha_images"

with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Load the trained neural network
model = load_model(MODEL_FILENAME)

# Grab some random CAPTCHA images to test against.
captcha_image_files = list(paths.list_images(CAPTCHA_IMAGE_FOLDER))
captcha_image_files = np.random.choice(captcha_image_files, size=(10,), replace=False)
y_test=[]
y_pred=[]
# loop over the image paths
for captcha_image_file in captcha_image_files:

    print(image_file)
    
    image1 = cv2.imread(image_file)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    image1 = cv2.copyMakeBorder(image1, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
    
    filename = os.path.basename(captcha_image_file)
    captcha_correct_text = os.path.splitext(filename)[0]
    y_test.append(filename)
    image = PIL.Image.open(CAPTCHA_IMAGE_FOLDER+"/"+filename).convert('1')
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
    letter_image_regions.append((1, 1, 30, 50))        
    letter_image_regions.append((30, 1, 50, 50))        
    letter_image_regions.append((50, 1, 70, 50))        
    letter_image_regions.append((70, 1, 100, 50))        
    letter_image_regions.append((100, 1, 125, 50)) 
    predictions = []       
    output = cv2.merge([image1] * 3)

    for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):

        x, y, w, h = letter_bounding_box


        letter_image=image.crop((x,y,w,h))
        letter_image.save("test.png","png")
        test_image = cv2.imread("test.png")
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        letter_image = resize_to_fit(test_image, 35, 35)

        letter_image = np.expand_dims(letter_image, axis=2)
        letter_image = np.expand_dims(letter_image, axis=0)
        prediction = model.predict(letter_image)

        # Convert the one-hot-encoded prediction back to a normal letter
        letter = lb.inverse_transform(prediction)[0]
        predictions.append(letter)

        # draw the prediction on the output image
        cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
        cv2.putText(output, letter, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        
    # Print the captcha's text
    captcha_text = "".join(predictions)
    print("CAPTCHA text is: {}".format(captcha_text))
    y_pred.append(captcha_text)

    cv2.imshow("Output", output)
    
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)    
Accuracy=str(round(len(y_pred[y_pred==y_test])/len(y_pred)*100,2)) +'%'
print(Accuracy)

import scikitplot as skplt
import matplotlib.pyplot as plt

skplt.metrics.plot_roc_curve(y_test, y_pred)
plt.show()
    #cv2.waitKey()