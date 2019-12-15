# deeplearning

# Tensorflow Object Detection for 3 Things:

            Car


   <p align="center">
  <img width="460" height="300" src="screenshots/ferari2.jpg">
</p>


        
        
          Person


<p align="center">
  <img width="460" height="300" src="screenshots/people.jpg">
</p>





         Bicycle


<p align="center">
  <img width="460" height="300" src="screenshots/cycle.jpg">
</p>




 # Main reference:

https://medium.com/object-detection-using-tensorflow-and-coco-pre/object-detection-using-tensorflow-and-coco-pre-trained-models-5d8386019a8


# reference code: for training and testing the model
https://github.com/tensorflow/models/blob/master/research/object_detection





## File videopretrained.py and imagepretrained.py is for testing the sample images and video using pretrained model


# Libraries needed to install:

tensorflow==1.4.0


opencv-python


keras==2.0.0


# environment needed to train:
            anaconda 3.5.0
https://www.anaconda.com/distribution/


# For labeliing each image by class you need:


[LabelImg GitHub link](https://github.com/tzutalin/labelImg)



The video will  abit slow to detect due to using fasterrcnn resnet model if run on local computer not cloud




# First Model and Second  Refer to:


https://1drv.ms/u/s!AjtR5zyBlsc9jxNXSR0e0kfWMUsa?e=c2oZdD

![Repo_List](screenshots/mod1.JPG)                                                           ![Repo_List](screenshots/mod2.JPG)
                














# PreTrained Refer to:


faster_rcnn_resnet50_coco



https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md



   ![Repo_List](screenshots/mod3.JPG)
   
   
   
   
   
 #                               Steps for Training the model
 
 
 
 
   # 1.Gathering data
                                
   https://www.kaggle.com/sanikamal/horses-or-humans-dataset
                        
   
   https://www.kaggle.com/tongpython/cat-and-dog
                                
                                
  #  2.Labeling data
[LabelImg GitHub link](https://github.com/tzutalin/labelImg)
               
 step a: run labelimg.py then label the images according to 3 classes mentioned
     
   <p align="center">
  <img width="460" height="300" src="screenshots/bus.JPG">
</p
                        
       
       
       
       
     step b: run python xml_to_csv.py to convert xml files that is labeled to csv files
     
     
            
   <p align="center">
  <img width="500" height="200" src="screenshots/xml.JPG">
</p
            



 
 
 
 
 
 
 
#
 
 
 
 
# 3.Generating TFRecords for training

<p align="center">
  <img width="500" height="400" src="screenshots/tf.PNG">
</p
                           
  
  
                                                  commands to generate tf records!!
  python generate_tfrecord.py — csv_input=images\train_labels.csv — image_dir=images\train — output_path=train.record
   python generate_tfrecord.py — csv_input=images\test_labels.csv — image_dir=images\test — output_path=test.record
        




#



                                 
  # 4.Configure Training by modifying parameters
  
  
  example:
  
  
  
  <p align="center">
  <img width="200" height="400" src="screenshots/finetune.JPG">
</p
  
  
                   
                   
                   
                                
        
  # 5.Training Model
                                
                             
                             
                             
                                
   # 6.Exporting inference graph
   
   
   
   
   
   # 7.Testing the object detector









