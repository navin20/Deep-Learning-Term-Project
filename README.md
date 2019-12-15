# deeplearning

# Tensorflow Object Detection for 3 Things:

            Car


   ![Repo_List](screenshots/ferari2.jpg)


        
        
          Person


![Repo_List](screenshots/people.jpg)





         Bicycle


![Repo_List](screenshots/cycle.jpg)




 # Main reference:

https://medium.com/object-detection-using-tensorflow-and-coco-pre/object-detection-using-tensorflow-and-coco-pre-trained-models-5d8386019a8


# reference code: for training and testing the model
https://github.com/tensorflow/models/blob/master/research/object_detection





## File videopretrained.py and imagepretrained.py is for testing the sample images and video using pretrained model


# Libraries needed to install:

tensorflow==1.4.0


opencv-python


keras==2.0.0


# For labeliing each image by class you need:


https://github.com/tzutalin/labelImg



The video will  abit slow to detect due to using fasterrcnn resnet model if run on local computer not cloud




# First Model and Second  Refer to:


https://1drv.ms/u/s!AjtR5zyBlsc9jxNXSR0e0kfWMUsa?e=c2oZdD


![Repo_List](screenshots/mod1.JPG)                                                  ![Repo_List](screenshots/mod2.JPG)







# PreTrained Refer to:


faster_rcnn_resnet50_coco



https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md



   ![Repo_List](screenshots/mod3.JPG)
   
   
   
   
   
 #                               Steps for Training the model
                                1.Gathering data
                                
   https://www.kaggle.com/sanikamal/horses-or-humans-dataset
                        
   
   https://www.kaggle.com/tongpython/cat-and-dog
                                
                                
                                2.Labeling data
                                
                                
                                3.Generating TFRecords for training
                                
                                 4.Configuring training
                                 
                                 5.Training model
                                6.Exporting inference graph
                                 7.Testing object detector









