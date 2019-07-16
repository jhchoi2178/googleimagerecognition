#image_loader.py

from PIL import Image

def url_to_rgb(image_id,PATH):
	file = Image.open(PATH+"/"+image_id+".jpg")
	r,g,b = np.array(file).T
	return r,g,b

def rgb_to_array(r,g,b):
	return r,g,b


def image_process(filename,width=1280,height=720):
	file = Image.open("data_train/"+filename+".jpg")
	file = file.resize((width,height),Image.ANTIALIAS)
	r,g,b = np.array(file).T/255.0
	return r,g,b