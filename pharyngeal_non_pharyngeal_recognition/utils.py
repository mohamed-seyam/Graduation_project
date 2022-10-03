from tensorflow import keras
from tensorflow.keras.metrics import Adam
from tensorflow.keras.layers import Input, Dense, Flatten, AveragePooling2D

def create_model_with_densenet():
    IMG_SHAPE = (224,224,3)

    # Create the base model from the pre-trained model MobileNet V2
    baseModel = tf.keras.applications.DenseNet121(include_top=False,weights='imagenet',input_tensor=Input(shape=(224, 224, 3)))
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(512, activation="relu")(headModel)
    headModel = Dense(1, activation="sigmoid")(headModel)


    model= Model(inputs=baseModel.input, outputs=headModel)

    for layer in baseModel.layers:
        layer.trainable = False

    # initialize the initial learning rate, number of epochs to train for,
    # and batch size
    INIT_LR = 1e-3
    EPOCHS = 25
    BS = 8
    # compile our model
    print("[INFO] compiling model...")
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    METRICS = [
                keras.metrics.TruePositives(name='tp'),
                keras.metrics.FalsePositives(name='fp'),
                keras.metrics.TrueNegatives(name='tn'),
                keras.metrics.FalseNegatives(name='fn'), 
                # keras.metrics.SpecificityAtSensitivity(1),
                # keras.metrics.SensitivityAtSpecificity(1),
                keras.metrics.BinaryAccuracy(name='accuracy'),
             
                keras.metrics.Precision(name='Percision'),
                keras.metrics.Recall(name='sensitiviy'),
                # keras.metrics.AUC(name='auc'),
                # keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
        ]
    model.compile(loss="binary_crossentropy", optimizer=opt,
        metrics= METRICS)

    return model