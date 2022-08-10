# MLP-classification
how use as classification:
MLP classification in python by skilearn Lib :
after denoising the pic or signal and..........we should get all feature from them
 then get them Label (calss1, class2 ,,,,,,,,,,,,,,,,)
after that use this Label as target 
**1.first with the numpy library  concatenate the class1 , class2,.........** 
this part give us an array with Nclass row 
**2.then use shuffle to mix the DATA** 
**3.fromfrom sklearn.model_selection import KFold** 
to use Kfold way as classification 
**4. from the skilearnneural_network import MLPClassifier** 
this  section let us to use the classification of svm
5.from sklearn.svm import SVC
