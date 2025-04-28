import cancer_ai_diagnoser_optimal_model_architecture


model_cancerDetector = cancer_ai_diagnoser_optimal_model_architecture.model_cancerDetector

DIAGNOSIS_MESSAGES = [ "Pneumonia detected", "Cancer detected", "Normal lungs detected" ]

def doOnlineInference_cancer (imagePath):
    test_data = []
    img = cancer_ai_diagnoser_optimal_model_architecture.cv2.imread(imagePath,0) #Replace plt.imread, with  gray scale cv2.imread(path,0), so that ui's image load process doesn't throw a pyimage10 error
    img = cancer_ai_diagnoser_optimal_model_architecture.cv2.resize(img, (cancer_ai_diagnoser_optimal_model_architecture.img_dims, cancer_ai_diagnoser_optimal_model_architecture.img_dims))
    img = cancer_ai_diagnoser_optimal_model_architecture.np.dstack([img, img, img])
    img = img.astype('float32') / 255
    test_data.append(img)
    prediction = model_cancerDetector.predict(cancer_ai_diagnoser_optimal_model_architecture.np.array(test_data))
    _prediction = round( prediction[0][0]*100, 3 )
    if ( _prediction > 50 ):
        _prediction = DIAGNOSIS_MESSAGES[1];
    elif ( _prediction < 50 ):
        _prediction = DIAGNOSIS_MESSAGES[2];  
    outputContent = _prediction + "\n"
    outputContent += "Raw Neural Network Output : " + str(prediction[0][0]) + ". A value closer to 1 signifies illness, while a value closer to 0 signifies normalness.\n\n"
    recordInferenceEvent (imagePath, outputContent)
    return outputContent



#Record each inference in a text file 
import datetime
def recordInferenceEvent ( imagePath, outputContent ):
    currentDate = datetime.datetime.now()
    with open("inference_record.txt", "a") as text_file:
        text_file.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        text_file.write("DATE/TIME : " + str(currentDate.month) + " " + str(currentDate.day) + ", " + str(currentDate.year) + "..." + str(currentDate.hour) + ":" + str(currentDate.minute) + ":" + str(currentDate.second) + "\n\n") 
        text_file.write("IMAGE : " + imagePath + "\n\n")
        text_file.write("RESULT : \n" + outputContent + "\n\n\n\n")

"""
NORMAL SAMPLES:
doOnlineInference_regularPneumonia("xray_dataset/val/NORMAL/NORMAL2-IM-1430-0001.jpeg")
doOnlineInference_regularPneumonia("xray_dataset/val/NORMAL/NORMAL2-IM-1427-0001.jpeg")

PNEUMONIA SAMPLES:
doOnlineInference_regularPneumonia("xray_dataset/val/PNEUMONIA/person1946_bacteria_4875.jpeg")
doOnlineInference_regularPneumonia("xray_dataset/val/PNEUMONIA/person1950_bacteria_4881.jpeg")

ACTUAL CORONAVIRUS SAMPLES:
doOnlineInference_cancer("coronavirus_positive_WeifangKong_et-al.jpg")
doOnlineInference_cancer("coronavirus_positive_day7_of_infection_UPSCALED.jpg")
"""
