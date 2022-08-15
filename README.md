# Test task for course

1. The model is a single LSTM block that takes as input word tokens and outputs predicted text readability.
   
   Command to lauch training:
   ```
   python train.py --num_epochs 30 --learning_rate 0.01
   ```
   You can download pretrained model here:
   https://drive.google.com/file/d/16tvXMRXk9Rj6SnBlag3vrNxpnt2rdEMY/view?usp=sharing
   
   Metrics for the model: rmse: 1.03
   
 2. To run the API, download pretrained model to the folder models/ (or train your own), and run the Flask API with:
    ```
    python service.py
    ```
 
 3. The service is deployed on Google Cloud.
    To test the script run the following commmand:
    ```
    python test_api.py
    ```
 
