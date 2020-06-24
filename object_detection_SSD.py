'''
Code Updated By: Sayak Banerjee

'''


'''Object Detection using SSD'''

# Importing the libraries
import torch
import cv2
from ssd_updated import build_ssd
import imageio
from data import BaseTransform, VOC_CLASSES as labelmap





'''Defining the function to perform Detections'''
def detect(frame, net, transform):                                                                                   # -> net - SSD network, transform - transforming the frame to make it compatible with the network
    height, width = frame.shape[0], frame.shape[1]                                                                          
    frame_t = transform(frame)[0]                                                                                    # -> Transforming the original frame to match the dimensions & requirements of the NN    
    x = torch.from_numpy(frame_t).permute(2, 0, 1)                                                                   # -> Convert Numpy Array to Torch Tensor. Permute used to change color channels from RBG(0,1,2) to GRB(2,0,1) as network was trained on that sequence.
    x = x.unsqueeze(0)                                                                                               # -> Expand the Dimensions to include the batch size
    y = net(x)                                                                                                       # -> Apply Network to model                                                                                            
    detections = y.data
    scale = torch.Tensor([width, height, width, height])
    
    # detections = [batch, number of classes, number of occurence, (score, x0, Y0, x1, y1)]
    
    for i in range(detections.size(1)):                                                                             # -> detections.size(1) = int(num_of_classes)
        j = 0
        while detections[0, i, j, 0] >= 0.6:                                                                        # -> confidence_score > 0.6
            pt = (detections[0, i, j, 1:] * scale).numpy()                                                          # -> Coordinates the detected object. Rectangles can drawn only on Numpy arrays
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2)                # -> Draw the Rectangle
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)  # -> Putting the Label 
            j += 1
    return frame





'''Creating the SSD neural network'''
net = build_ssd('test')
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage))




'''Creating the transformation'''
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))




'''Doing some Object Detection on a video'''
reader = imageio.get_reader('funny_dog.mp4')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('output.mp4', fps = fps)

for i, frame in enumerate(reader):
    frame = detect(frame, net.eval(), transform)
    writer.append_data(frame)
    print(i)
    
writer.close()