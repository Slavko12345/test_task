# Test task for course

1. The model used is a single LSTM block that takes as input word tokens and outputs predicted text readability.
   
   Command to lauch training:
   ```
   python train.py --num_epochs 30 --learning_rate 0.01
   ```
   You can download pretrained model here:
   https://drive.google.com/file/d/16tvXMRXk9Rj6SnBlag3vrNxpnt2rdEMY/view?usp=sharing
   
   Metrics for the model: rmse: 1.03
   
 2. Script for running the Flask API:
 ```
 python service.py
 ```
 
 3. The service is deployed on Google Cloud.
 
    Script for testing the request to API:
 ```
 python test_api.py
 ```
 
