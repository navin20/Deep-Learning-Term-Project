# deeplearning

Tensorflow Object Detection for 3 Things:

                        Car
                        ![alt text](https://www.google.co.th/search?hl=en&authuser=0&biw=1366&bih=625&tbm=isch&sxsrf=ACYBGNR6rgspcmx35UxFcJQsTvNrv1lzZg%3A1576390303994&sa=1&ei=n871XbCrPMXG4-EP7uiusA0&q=person&oq=person&gs_l=img.12...0.0..92796...0.0..0.0.0.......0......gws-wiz-img.RHIQJWxl_BA&ved=0ahUKEwjw9uLA_7bmAhVF4zgGHW60C9YQ4dUDCAc)
    
                        Person
                        
                        
    
                        Bicycle


Main reference:

https://medium.com/object-detection-using-tensorflow-and-coco-pre/object-detection-using-tensorflow-and-coco-pre-trained-models-5d8386019a8


reference code: for training and testing the model
https://github.com/tensorflow/models/blob/master/research/object_detection


File videopretrained.py and imagepretrained.py is for testing the sample images and video using pretrained model


Libraries needed to install:

tensorflow==1.4.0


opencv-python


keras==2.0.0


For labeliing each image by class you need:


https://github.com/tzutalin/labelImg



The video will  abit slow to detect due to using fasterrcnn resnet model if run on local computer not cloud




