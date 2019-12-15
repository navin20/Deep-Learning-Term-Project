#                     Deeplearning Term-Project




#    This project is about using resnet-50(Residual Neural Network) as a  feature extractor   with the help of tensorflow to help in detecting specific objects using transfer learning techniques






##                   Resnet-50
                 The ResNet-50 model consists of 5 stages each with a convolution and Identity
             block. Each convolution block has 3 convolution layers and each identity block
             ResNet-50 is a convolutional neural network that is trained on more than a million images from the ImageNet database . The network is 50 layers deep and can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals. As a result, the network has learned rich feature representations for a wide range of images. The network has an image input size of 224-by-224.
   <p align="center">
  <img width="785" height="202" src="screenshots/resnet.JPG">
</p








#                      [Using resnet model to detect Objects With the Help of tensorflow api ](https://www.tensorflow.org/)






 #    Tensorflow Object Detection for 3 Things:

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




 # Main reference for steps how to do:

 [Steps](https://github.com/navin20/deeplearning#Steps-for-Training-the-model)


# Main reference code: for training and testing the model
[Code](https://github.com/navin20/deeplearning#)





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
   # data is not really from any high source or website with datasets it is random images from many searching websites
                                
  [datasets](https://github.com/navin20/Deep-Learning-Term-Project/tree/master/datasets)
                                
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
   
## here data is converted from csv to tf records which basically turns the csv based on the class and the order mapped to the class in generate_tfrecords.py and also should be same as labelmap.pbtxt
        




#



                                 
  # 4.Configure Training by modifying parameters
  
  
 
 
 example:
  
  
 
 
 
 ## Fine tune the model to the pretrained model by using  transfer learning technique = as i said before this part using the pre trained mention below with freezing the last few layers and train the last few layers or 1 layer to make the current object detector  be familiar with the the pre trained model classes and what detects what since it has million of datas based on the coco datasets and many 1000 classes.
  
  
  
  
  
  
  <p align="center">
  <img width="739" height="47" src="screenshots/finetune.JPG">
</p
            
  
  
## create labelmap.pbtxt to map class numbers with the class name needed for training process
and the mapping of number should be very similar to generate_tfrecords.py file
            
   
<p align="center">
<img width="113" height="253" src="screenshots/labelmap.JPG">
</p
            
            
  
  
                   
                   
#

                                
        
  # 5.Training Model
  
             train by command:
                 
            python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_resnet50_coco.config
                                   
                                   
                                   
                                   
                                   
  ## Here the training steps really begins before was just data preprocessing labelling and modifying the model and config files.
  
  
## Each step of training reports the loss, eventually it will come down as training progresses. It’s recommend to allow your model to train until the loss consistently drops below 0.05.                                 
  
  
  
  Model 1 600 epoch with 50 images
  ## this model is pretty good with close objects and sometimes detect very good with them
  
  
  <p align="center">
  <img width="653" height="305" src="screenshots/epoch1.JPG">
</p
            
            
            
  <p align="center">
  <img width="607" height="261" src="screenshots/epoch2.JPG">
</p
  
  #
  
## this  model works best with object being detected very far and is has more images of people or objects being detected far
 

#

#
            
            
            
  
  
  
                                
                             
                             
                             
                                
   # 6.Exporting inference graph
   
   ## this is the part where the training ends and is exported as a form of frozen graph to help in detection after the train stops
   
   ## commands:
   python export_inference_graph.py  --input_type image_tensor  --pipeline_config_path training/faster_rcnn_resnet50_coco.config --trained_checkpoint_prefix training/model.ckpt-600 --output_directory inference_graph
   
   
   
   
   <p align="center">
  <img width="607" height="261" src="screenshots/frozen.JPG">
</p
            
            
            
##
   
   
   
   
   
   
   
   # 7.Testing the object detector
   
   
 ## try testing the  model with images or videos as an example with object_detection_video.py and object_detection_image.py
 
 
 ## Here is an example of comparison of 3 models
 
  ### Pretrained Model
  
  
 <p align="center">
  <img width="607" height="261" src="screenshots/pretrained.JPG">
</p

#
            
            
  ### Model 1
  
  
   <p align="center">
  <img width="607" height="261" src="screenshots/model1.JPG">
</p
     
  
  
  
#  
  
  
  
  ### Model 2
  
  
 <p align="center">
  <img width="607" height="261" src="screenshots/model2.JPG">
</p                       
  
  
       
       
#      
       
       
# Other References

##            API:
## https://github.com/tensorflow/models/tree/master/research/object_detection

## How to use Transfer Learning

## https://machinelearningmastery.com/transfer-learning-for-deep-learning


 










