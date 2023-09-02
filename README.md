# AUC-Image-Tagging-App

# Table of contents
1. [Welcome](#welcome)
2. [How to install](#install)
3. [How to run & use](#runuse)
4. [Technologies used](#tech)
5. [Research and Additional Notes](#research)

## Welcome <a name="welcome"></a>
Hi everyone! Welcome to the "AUC Image Tagging App." This app has been developed by two undergraduate students at the American University in Cairo (AUC) to help the AUC librarians in their tagging process for the records at the library. This app is intended to be used by authorized AUC librarians only, so if you're not an AUC librarian, you won't have access to the Google Drive folder that the app is dependent upon, and you won't be able to use the app.

## How to install <a name="install"></a>
To run our app, you first need to install some dependencies. Just follow the steps below, and everything should run smoothly.
1. Install python3 from: [https://www.python.org/downloads/](https://www.python.org/downloads/).
2. Download the code as a zip and then unzip it.
3. Go to the directory where the code was saved.
4. Right-click and click on the "Run in Terminal" option. If you do not find this option, open the command prompt and type `cd` followed by the directory in which the file was downloaded.
![image](https://user-images.githubusercontent.com/78408934/187026405-b51781b7-e9ea-4fd3-b5cf-6f99db5ab736.png)
5. Type `pip install -r Requirements.txt` in the terminal.
6. Copy and paste the contents of "Run.txt" into the terminal.
7. Ctrl + click on the outputed link, which will take you to the website.
![image](https://user-images.githubusercontent.com/78408934/187026498-236c9638-511d-4b8e-9f22-eb15c60b5c07.png)

## How to run & use <a name="runuse"></a>
To run the app, simply double-click on the ImageTag.py file.

### User Guide
To make the most of the "AUC Image Tagging App," follow these steps:

1. Click on the "Upload Image" button on the screen's top left.

2. Choose the photo from your PC and click okay.

3. Tags are automatically generated on the right. To select a tag, just click on it. To unselect a tag, click on it once more.

4. Choose your tags from the generated tags on the right.

5. If you want to add tags that are not available, click on the "Manual Tags" button at the bottom left and follow the instructions.

6. Manual tags will be added to the list and automatically selected. They can be unselected too.

7. Press the "Store" button and wait until your image is uploaded. Once the image is uploaded, the tags list will be responsive to the mouse's hovering over it.

8. Finally, you will find the image and the list of tags in Arabic and English uploaded to the following drive link: [https://drive.google.com/drive/folders/1VxcpHDDE7Psdyvfm_tcBBQ_RY2D6Snf2](https://drive.google.com/drive/folders/1VxcpHDDE7Psdyvfm_tcBBQ_RY2D6Snf2)

## Technologies used <a name="tech"></a>
The "AUC Image Tagging App" leverages a variety of cutting-edge technologies to streamline the tagging process and enhance efficiency. Our team has carefully chosen and integrated the following technologies into the app's development:

1. [Pytorch](https://pytorch.org/): A deep learning framework to build and train models.

2. [PyQT5](https://riverbankcomputing.com/software/pyqt/): A front-end framework based on Python to design our app.

3. [Inception-v4](https://arxiv.org/abs/1602.07261): A convolutional neural network architecture that builds on previous iterations of the Inception family by simplifying the architecture and using more inception modules than Inception-v3.

4. CSP ResNet: A convolutional neural network where we apply the Cross Stage Partial Network (CSPNet) approach to ResNet. The CSPNet partitions the feature map of the base layer into two parts and then merges them through a cross-stage hierarchy. The use of a split and merge strategy allows for more gradient flow through the network.

5. Google Drive and Google Drive API.

## Research and Additional Notes <a name="research"></a>
### Research Efforts
During our research and development process, we explored various approaches and considered different methodologies to create an efficient image tagging solution. Here are some key insights from our research:

- Initially, we experimented with building deep learning models using reinforcement learning techniques. While this approach showed the potential for generating better results, it was clear that it would require significantly more computational resources and time than what was available for this project. Consequently, we decided to explore alternative solutions that could be implemented more feasibly.

- Subsequently, we opted for traditional Convolutional Neural Network (CNN) models to achieve our tagging objectives. To enhance their accuracy, we integrated the results of two models and removed duplicate tags before presenting them to the user, ensuring a more refined tagging process.

- We also considered the use of [The All-Seeing model](https://www.marktechpost.com/2023/08/10/breakthrough-in-the-intersection-of-vision-language-presenting-the-all-seeing-project/). This model showcased promising results and advanced techniques, but it was not fully developed at the time of creating this application. If you are interested in further improving our app, you may explore integrating The All-Seeing model into the system for enhanced tagging capabilities.

- Initially, we contemplated hosting our app online and utilizing JavaScript in conjunction with HTML and CSS to create a more sophisticated front-end. However, after careful consideration, we chose the PyQT framework as it provided an efficient front-end solution and allowed us to allocate more time to model development and training.

### Data Considerations
It's essential to note that we did not have access to the specific library images during our model training process. Instead, we trained the models on large sets of random images. If you find that the models require improvements for better and more accurate tagging, we recommend retraining them using the library's tagged photos for optimal results.

## Additional Notes
...
