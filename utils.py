import numpy as np
import pandas as pd 
import tensorflow as tf 
import cv2


def mkdir_ifnotexists(dir):
    if not Path(dir).exists():
        Path(dir).mkdir(parents=True, exist_ok=True)
    else:
        print(f'{dir} Directory Already Exists')

def video2frames(video_dir, video_name, frames_dir):
    cmd = "ffmpeg -i %s -start_number 0 -vsync 0 %s/%s_%%05d.png" % (f'{video_dir}', frames_dir, video_name)
    os.system(cmd)


# ====================================================================== #
#   this class is to be used by specific Qthread, it's considered a long Task
# ====================================================================== #
class OpticalFlow(QObject):

    def __init__(self, video_dir, frames_names, output_dir=opticalFlow_dir):
        super().__init__()
        self.video_dir = video_dir
        self.frames_names = frames_names
        self.output_dir = output_dir
        mkdir_ifnotexists(opticalFlow_dir)

    def calculate(self, ):
        """ Long Running Task """
        # open video capture
        self.cap = cv2.VideoCapture(self.video_dir)

        # get total number of video frames
        num_frames = len(self.frames_names)

        curr_frame_idx = 0

        # read the first frame
        ret, previous_frame = self.cap.read()
        # proceed if frame reading was successful
        if ret:
            # resize frame
            frame = cv2.resize(previous_frame, (400, 600))

            # convert to gray
            previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # create hsv output for optical flow
            hsv = np.zeros_like(frame, np.float32)

            # set saturation to 1
            hsv[..., 1] = 1.0

            # loop over all the frames
            while True:

                # capture frame-by-frame
                ret, frame = self.cap.read()

                # Termination condition, when No frame is left to read, EXIT
                if not ret:
                    break

                # resize frame
                frame = cv2.resize(frame, (400, 600))

                # convert to gray
                current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # calculate optical flow
                # flow = cv2.calcOpticalFlowFarneback(previous_frame, current_frame, None, 0.5, 5, 15, 3, 5, 1.2, 0,)

                # print("Hi")
                # =========================================================
                ## Normal TVL1
                dtvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
                flow = dtvl1.calc(previous_frame, current_frame, None)
                
                # flow = ImgBound(flow, 20)
                # ==========================================================
                # print("Bay")

                # convert from cartesian to polar coordinates to get magnitude and angle
                magnitude, angle = cv2.cartToPolar(
                    flow[..., 0], flow[..., 1], angleInDegrees=True,
                )

                # set hue according to the angle of optical flow
                hsv[..., 0] = angle * ((1 / 360.0) * (180 / 255.0))

                # set value according to the normalized magnitude of optical flow
                hsv[..., 2] = cv2.normalize(
                    magnitude, None, 0.0, 1.0, cv2.NORM_MINMAX, -1,
                )

                # multiply each pixel value to 255
                hsv_8u = np.uint8(hsv * 255.0)

                # convert hsv to bgr
                bgr = cv2.cvtColor(hsv_8u, cv2.COLOR_HSV2BGR)

                # update previous_frame value
                previous_frame = current_frame

                # store the result as an image png
                cv2.imwrite(self.output_dir + self.frames_names[curr_frame_idx], bgr)

                # move to the next frame
                curr_frame_idx +=1

                # report progress
                print("Finished frame :{} ".format(curr_frame_idx))
        
        self.cap.release()

def check_overlap(df_train, df_test, df_valid):
    train_subjects = [x.split("_")[0] for x in df_train["image"]]
    print(len(train_subjects))
    print("unique train subjects:", f"{len(set(train_subjects))}")
    val_subjects = [x.split("_")[0] for x in df_valid["image"]]
    print(len(val_subjects))
    print("unique val subjects:", f"{len(set(val_subjects))}")
    test_subjects = [x.split("_")[0] for x in df_test["image"]]
    print(len(test_subjects))
    print("unique test subjects:", f"{len(set(test_subjects))}")
    if len(set(train_subjects) - set(val_subjects)) == len(set(train_subjects)):
        print("no overlap between train and validate")
    else:
        print(f"subject overlap: {len(set(val_subjects) - set(train_subjects))} : {set(val_subjects) - set(train_subjects)}") 
    if len(set(train_subjects) - set(test_subjects)) == len(set(train_subjects)):
        print("no overlap between train and test")
    else:
        print(f"subject overlap: {len(set(train_subjects) - set(test_subjects))} : {set(test_subjects) - set(train_subjects)}")    
    if len(set(test_subjects) - set(val_subjects)) == len(set(test_subjects)):
        print("no overlap between validate and test") 
    else:
        print(f"subject overlap: { len(set(test_subjects) - set(val_subjects))} : { set(test_subjects) - set(val_subjects)}")   


def train_generator(df, image_dir, x_col, y_col, shuffle=True, batch_size= 64, seed=1, target_w = 320, target_h = 320):
    """
    Return generator for training set, normalizing using batch
    statistics.

    Args:
      train_df (dataframe): dataframe specifying training data.
      image_dir (str): directory where image files are held.
      x_col (str): name of column in df that holds filenames.
      y_cols (str) : name of column in df that hold label of each image.
      sample_size (int): size of sample to use for normalization statistics.
      batch_size (int): images per batch to be fed into model during training.
      seed (int): random seed.
      target_w (int): final width of input images.
      target_h (int): final height of input images.
    
    Returns:
        train_generator (DataFrameIterator): iterator over training set
    """        
    print("getting train generator...") 
    # normalize images 
    image_generator = ImageDataGenerator(
        samplewise_center=True,                      # make mean = 0 to every sample cuz train data will be very big
        samplewise_std_normalization= True)
    
    # flow from directory with specified batch size
    # and target image size
    generator = image_generator.flow_from_dataframe(
            dataframe=df,
            directory=image_dir,
            x_col=x_col,          # the name of the image 
            y_col=y_col,          # the class of the image
            class_mode="raw",     
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            target_size=(target_w,target_h),   # rescaling the images 
            color_mode="grayscale")
    
    return generator

def test_and_valid_generator(valid_df, test_df , train_df, image_dir, x_col, y_col, sample_size=1000, batch_size=8, seed=1, target_w = 320, target_h = 320):
    """
    Return generator for validation set and test test set using 
    normalization statistics from training set.    

    Args:
      valid_df (dataframe): dataframe specifying validation data.
      test_df (dataframe): dataframe specifying test data.
      train_df (dataframe): dataframe specifying training data.
      image_dir (str): directory where image files are held.
      x_col (str): name of column in df that holds filenames.
      y_cols (str): name of column in df that holds the label of the image.
      sample_size (int): size of sample to use for normalization statistics.
      batch_size (int): images per batch to be fed into model during training.
      seed (int): random seed.
      target_w (int): final width of input images.
      target_h (int): final height of input images.
    
    Returns:
        test_generator (DataFrameIterator) and valid_generator: iterators over test set and validation set respectively
    """
    print("getting train and valid generators...")
    # get generator to sample dataset
    raw_train_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=train_df, 
        directory=image_dir, 
        x_col=x_col, 
        y_col=y_col, 
        class_mode="raw", 
        batch_size=sample_size, 
        shuffle=True, 
        target_size=(target_w, target_h),
        color_mode="grayscale"
        )
    
    # get data sample
    batch = raw_train_generator.next()   # to iterator to save all the data into batche
    data_sample = batch[0]               # take batch[0] to use it to normalize the dev and test datasets

    # use sample to fit mean and std for test set generator
    image_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization= True)
    
    # fit generator to sample from training data
    image_generator.fit(data_sample)

    # get test generator
    valid_generator = image_generator.flow_from_dataframe(
            dataframe=valid_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_col,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h),
            color_mode="grayscale")

    test_generator = image_generator.flow_from_dataframe(
            dataframe=test_df,
            directory=image_dir,
            x_col=x_col,   
            y_col=y_col,   
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h), # 
            color_mode="grayscale")

    return valid_generator , test_generator  

def split_data(data , train_n = .8 ):
    ''' This function is used to split data into train, test, valid 
    Args:
        data: dataframe of all data
        train_n : percentage of the train data default is .8
    Return:
        train_df: dataframe contain train data 
        valid_df: dataframe contain valid data 
        test_df : dataframe contain test data 
    '''
    #data = data.sample(frac=1, random_state = 0 ).reset_index(drop = True , )  #randomize dataframe and take a sampls = total number of rows
    t_split = int(data.shape[0]*train_n)   # split value for the train
    v_split = int((data.shape[0] - t_split)/2)  + t_split  # split value for valid
    df_train = data.iloc[:t_split,:]
    df_valid = data.iloc[t_split:v_split,:]
    df_test  = data.iloc[v_split: ,:]



    return df_train, df_valid, df_test


def true_positives(y, pred):
    """
    Count true positives.

    Args:
        y (np.array): ground truth, size (n_examples)
        pred (np.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        TP (int): true positives
    """
    TP = 0
    
    # get thresholded predictions
    # thresholded_preds = pred >= th

    # compute TP
    TP = np.sum((y == 1) & (pred == 1))
    
    return TP

def true_negatives(y, pred):
    """
    Count true negatives.

    Args:
        y (np.array): ground truth, size (n_examples)
        pred (np.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        TN (int): true negatives
    """
    TN = 0
    
    # get thresholded predictions
    # thresholded_preds = pred >= th


    
    # compute TN
    TN = np.sum((y == 0) & (pred == 0 ))
    
    return TN

def false_positives(y, pred):
    """
    Count false positives.

    Args:
        y (np.array): ground truth, size (n_examples)
        pred (np.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        FP (int): false positives
    """
    FP = 0
    
    # get thresholded predictions
    # thresholded_preds = pred >= th


    # compute FP
    FP =  np.sum((y == 0) & (pred == 1 ))
    
    
    return FP

def false_negatives(y, pred):
    """
    Count false positives.

    Args:
        y (np.array): ground truth, size (n_examples)
        pred (np.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        FN (int): false negatives
    """
    FN = 0
    
    # get thresholded predictions
    # thresholded_preds = pred >= th


    
    # compute FN
    FN =np.sum((y == 1) & (pred == 0 ))
    
    
    return FN

def get_accuracy(y, pred):
    """
    Compute accuracy of predictions at threshold.

    Args:
        y (np.array): ground truth, size (n_examples)
        pred (np.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        accuracy (float): accuracy of predictions at threshold
    """
    accuracy = 0.0
    

    
    # get TP, FP, TN, FN using our previously defined functions
    TP = true_positives(y,pred)
    FP = false_positives(y,pred)
    TN = true_negatives(y,pred)
    FN = false_negatives(y,pred)

    # Compute accuracy using TP, FP, TN, FN
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    
    ### END CODE HERE ###
    
    return accuracy


def get_prevalence(y):
    """
    Compute prevalence.

    Args:
        y (np.array): ground truth, size (n_examples)
    Returns:
        prevalence (float): prevalence of positive cases
    """
    prevalence = 0.0 
    prevalence = np.sum(y==1)/ len(y)   
    return prevalence

def get_sensitivity(y, pred):
    """
    Compute sensitivity of predictions at threshold.
    
    Recall

    Args:
        y (np.array): ground truth, size (n_examples)
        pred (np.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        sensitivity (float): probability that our test outputs positive given that the case is actually positive
    """
    sensitivity = 0.0
    
  
    
    # get TP and FN using our previously defined functions
    TP = true_positives(y,pred)
    FN = false_negatives(y,pred)

    # use TP and FN to compute sensitivity
    sensitivity = TP/(TP+FN)
    
    return sensitivity

def get_specificity(y, pred):
    """
    Compute specificity of predictions at threshold.

    Args:
        y (np.array): ground truth, size (n_examples)
        pred (np.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        specificity (float): probability that the test outputs negative given that the case is actually negative
    """
    specificity = 0.0
    
    
    # get TN and FP using our previously defined functions
    TN = true_negatives(y,pred)
    FP = false_positives(y,pred)
    
    # use TN and FP to compute specificity 
    specificity = TN/(TN+ FP)

    return specificity

def get_ppv(y, pred):
    """
    Compute PPV of predictions at threshold.
    
    Precision

    Args:
        y (np.array): ground truth, size (n_examples)
        pred (np.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        PPV (float): positive predictive value of predictions at threshold
    """
    PPV = 0.0

    
    # get TP and FP using our previously defined functions
    TP = true_positives(y,pred)
    FP = false_positives(y,pred)

    # use TP and FP to compute PPV
    PPV = TP/(TP+FP)
 
    return PPV

def get_npv(y, pred):
    """
    Compute NPV of predictions at threshold.

    Args:
        y (np.array): ground truth, size (n_examples)
        pred (np.array): model output, size (n_examples)
        th (float): cutoff value for positive prediction from model
    Returns:
        NPV (float): negative predictive value of predictions at threshold
    """
    NPV = 0.0
    
    
    # get TN and FN using our previously defined functions
    TN = true_negatives(y,pred)
    FN = false_negatives(y,pred)

    # use TN and FN to compute NPV
    NPV = TN/(TN+FN)
    
    return NPV