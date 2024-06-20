# Neural_Style_Transfer
**OVERVIEW**

Neural Style Transfer is a computational technique in that uses deep learning to merge the content of one image with the style of some another image. It uses a pre-trained model, called VGG19, to extract content features and style features. It optimizes the required target image iteratively with content image's features and match the statistical properties(such as correlations between different feaure maps) derived from style image. it helps generate visually appealing images that blend the asrtisitc style of one image with that of the another

**INSTALLATION INSTRUCTIONS**  
*Must have Python installed in the system  
*Must have pip to manage the packages  

**Necassary Libraries**
*Streamlit
*Torch
*Torchvision
*Pillow   

**How to Run the Code**
Start the Streamlit Application by giving this command-"streamlit run app.py". Choose and upload the Content and Style Images.The image will be loaded-resized,get converted to tensor for feature extraction and normalized, all of which has been included in the load_image function. The target image will be initialized to the content image itself. The content loss and style loss have been initialized to 0. Content loss measures the difference in content between images, while style loss measures differences in texture and patterns using Gram matrices. Total loss combines content loss and style loss, using alpha and beta. The values for all the hyperparameters have been taken arbitirarily.The streamlit UI sets up a web-based interface using Streamlit (app.py) where users can upload content and style images, visualize them, and see the generated image after style transfer. The Adam optimizer then minimizes the total loss. The features for the content and style images to calculate all the losses have been extracted using VGG19-a pre-trained model on Imagenet. Finally, the users can interact with the application using the interface, uploading images, initiating the NST process and viewing the final generated image.


