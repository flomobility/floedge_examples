### Update absolute paths
Open ```utils/config.py``` and add the complete absolute path to ```h264_folder```, ```log_dir```, and ```countfile```
Update the complete absolute path in ```security_system/securitycam.py``` script, ```line 8```

### Update emails of sender and recepients
In order to send mails, you will need to add your email ID and password to ```security_system/confidential.txt``` file. To do this securely, a temporary app password can be generated that you can use while running the project and delete when not in use. Follow the steps [here](https://support.google.com/accounts/answer/185833?hl=en) to generate the 16 digit temporary password.

Open ```security_system/confidential.txt```:
```
{"recepients": ["sampleemail@abc.com"], "myemail": "youremail@abc.com", "mypass": "yourpassword"}
```
Replace ```sampleemail@abc.com``` with a valid email ID. You can add multiple recepient emails seperated by commas within the list. Replace ```youremail@abc.com``` with your email ID and ```yourpassword``` with the 16 digit temporary password that is generated in the above step.

### Running the example
Run the ```securitycam.py``` script
```
cd security_system/
python securitycam.py
```
