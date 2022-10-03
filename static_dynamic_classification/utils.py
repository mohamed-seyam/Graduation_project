import tensorflow as tf
from tensorflow.keras.metrics import RMSprob
import os 


def create_model():
    '''This is function used to create the model reponsible for classification of frames 
    whether they are static or dyanmic 
    Args:
        No Args 
    Returns:
        model build
    '''
    model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, (3,3) ,activation = 'relu' , input_shape = (128 , 128 , 1) ) , 
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(32,(3,3),activation = 'relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64,(3,2), activation = 'relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64,activation = 'relu'),
            tf.keras.layers.Dense(1,activation = "sigmoid")
        ])

    model.compile(loss = "binary_crossentropy", optimizer = RMSprop(learning_rate = .001),
                    metrics = ['accuracy'])
    
    return model


def compute_class_frq(df):
  '''This function used to compute number of postive versus negative image
  Args:
    df(dataframe):dataframe contain classes
  Return:
    pos: int number indicate number of positive classes 
    neg: int number indicate number of negative classes
  '''
  pos, neg = df["class"].value_counts() 
  pos = pos /df.shape[0]
  neg = neg/df.shape[0]

  return pos,neg



def labeling(pharyngeal_images,target_file):
    if os.path.exists(target_file):
        os.remove(target_file)
        print("old file deleted sucessfully")
    else:   
        print("Creating output file")
        target_file = open(target_file,"w",newline="")
        fieldnames = ["image","class"]
        writer_file = csv.DictWriter(target_file,fieldnames=fieldnames)
        writer_file.writeheader()
      
    imgs = os.listdir(images_dir)
    imgs = [img.lower() for img in imgs]
    pharyngeal = [item for sublist in pharyngeal_generate for item in sublist]
    non_pharyngeal = list(set(imgs) - set(pharyngeal))
    
    zip0 = list(zip(non_pharyngeal, cycle('0')))
    zip1 = list(zip(pharyngeal, cycle('1')))
    
    zipped = zip0 + zip1
    sorted_zipped = sorted(zipped, key = lambda x: x[0])
    
    count = len(sorted_zipped)
    for i in range(count):
        row = {"image": str(sorted_zipped[i][0])  ,"class": int(sorted_zipped[i][1])}  
        writer_file.writerow(row) 