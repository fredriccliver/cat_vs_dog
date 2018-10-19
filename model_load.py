## Keras - 모델 저장하고 불러오기 
## https://3months.tistory.com/150

#%%
import os
work_path = os.path.dirname(os.path.abspath('__file__'))
work_path

#%%
from keras.models import model_from_json
json_file = open(work_path + "/model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")
print("model loaded.")


loaded_model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['accuracy'])

# model evaluation
score = loaded_model.evaluate(X,Y,verbose=0)

print("%s : %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))