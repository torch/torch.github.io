---
layout: post
title: Deep Fun with OpenCV and Torch
comments: True
author: egor-burkov
excerpt: In this post, we'll have some fun with OpenCV 3.0 + Torch to build live demos of Age & Gender Classification, NeuralStyle, NeuralTalk and live Image classification.
picture: https://raw.githubusercontent.com/torch/torch.github.io/master/blog/_posts/images/opencv_age_small.png
---

<!---# Deep Fun with OpenCV and Torch-->

The [OpenCV](http://opencv.org/) library implements tons of useful image processing and computer vision algorithms, as well as the high-level GUI API. Written in C++, it has bindings in Python, Java, MATLAB/Octave, C#, Perl and Ruby. We present the Lua bindings that are based on Torch, made by [VisionLabs](http://visionlabs.ru).

By combining OpenCV with scientific computation abilities of Torch, one gets an even more powerful framework capable of handling computer vision routines (e.g. face detection), interfacing video streams (including cameras), easier data visualization, GUI interaction and many more. In addition, most of the computationally intensive algorithms are available on GPU via [cutorch](https://github.com/torch/cutorch). All these features may be essentially useful for those dealing with deep learning applied to images.

Usage Examples
===

### Live Image Classification

A basic example may be live CNN-based image classification. In the following demo, we grab a frame from the webcam, then take a central crop from it and use a small ImageNet classification pretrained network to predict what's in the picture. Afterwards, the image itself and the 5 most probable class names are displayed.

[![ImageNet classification demo](https://cloud.githubusercontent.com/assets/9570420/14849851/6982c4de-0c86-11e6-80c5-d7c4cc8a0f3d.png)](http://cdn.makeagif.com/media/2-28-2016/p4xoRF.gif)

The comments should explain the code well. *Note: this sample assumes you already have the trained CNN; see the [original code on GitHub](https://github.com/szagoruyko/torch-opencv-demos/blob/master/imagenet_classification/demo.lua) by Sergey Zagoruyko that automatically downloads it.*

```lua
local cv = require 'cv'
require 'cv.highgui' -- GUI
require 'cv.videoio' -- Video stream
require 'cv.imgproc' -- Image processing (resize, crop, draw text, ...)
require 'nn'

local capture = cv.VideoCapture{device=0}
if not capture:isOpened() then
   print("Failed to open the default camera")
   os.exit(-1)
end

-- Create a new window
cv.namedWindow{winname="Torch-OpenCV ImageNet classification demo", flags=cv.WINDOW_AUTOSIZE}
-- Read the first frame
local _, frame = capture:read{}

-- Using network in network http://openreview.net/document/9b05a3bb-3a5e-49cb-91f7-0f482af65aea
local net = torch.load('nin_nobn_final.t7'):unpack():float()
local synset_words = torch.load('synset.t7', 'ascii')

-- NiN input size
local M = 224

while true do
   local w = frame:size(2)
   local h = frame:size(1)

   -- Get central square crop
   local crop = cv.getRectSubPix{frame, patchSize={h,h}, center={w/2, h/2}}
   -- Resize it to 256 x 256
   local im = cv.resize{crop, {256,256}}:float():div(255)
   -- Subtract channel-wise mean
   for i=1,3 do
      im:select(3,i):add(-net.transform.mean[i]):div(net.transform.std[i])
   end
   -- Resize again to CNN input size and swap dimensions
   -- to CxHxW from HxWxC
   -- Note that BGR channel order required by ImageNet is already OpenCV's default
   local I = cv.resize{im, {M,M}}:permute(3,1,2):clone()

   -- Get class predictions
   local _, classes = net:forward(I):view(-1):float():sort(true)

   -- Caption the image
   for i=1,5 do
      cv.putText{
         crop,
         synset_words[classes[i]],
         {10, 10 + i * 25},
         fontFace=cv.FONT_HERSHEY_DUPLEX,
         fontScale=1,
         color={200, 200, 50},
         thickness=2
      }
   end

   -- Show it to the user
   cv.imshow{"Torch-OpenCV ImageNet classification demo", crop}
   if cv.waitKey{30} >= 0 then break end

   -- Grab the next frame
   capture:read{frame}
end
```

### Live Age and Gender Prediction

A more interesting demonstration can be made with CNNs [described here](http://www.openu.ac.il/home/hassner/projects/cnn_agegender/) trained to predict age and gender by face. Here, we take a frame from live stream, then use the popular [Viola-Jones cascade object detector](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.10.6807) to extract faces from it. Such detectors ship with OpenCV pretrained to detect faces, eyes, smiles etc. After the faces have been found, we basically crop them and feed to CNNs which yield age and gender predictions data. The faces in the image are marked with rectangles, which are then labeled by predicted age and gender.

Detecting faces and drawing the results from Torch for a single image is this easy:

```lua
require 'cv.objdetect'
local faceDetector = cv.CascadeClassifier{'haarcascade_frontalface_default.xml'}
local faces = faceDetector:detectMultiScale{image}

for i=1,faces.size do
   local f = faces.data[i]
   cv.rectangle{image, {f.x, f.y}, {f.x+f.w, f.y+f.h}, color={255,0,255,0}}
end
```

Of course, this is quite an inefficient way of face detection provided just to become a simple sample. For example, it could involve tracking techniques.

The entire code is [available on GitHub](https://github.com/szagoruyko/torch-opencv-demos/blob/master/age_gender/demo.lua). Here's a sample from IMAGINE Lab by Sergey Zagoruyko:

![Age & Gender Demo](https://cloud.githubusercontent.com/assets/4953728/12299217/fc819f80-ba15-11e5-95de-653c9fda9b83.png)

And here is a heavy just-for-fun GIF:

[![And here is is a heavy just-for-fun GIF](https://cloud.githubusercontent.com/assets/9570420/14849849/698022b0-0c86-11e6-82f2-452b343c786c.png)](http://cdn.makeagif.com/media/3-11-2016/afVDJO.gif)

### NeuralTalk2

A good image captioning example is [NeuralTalk2](https://github.com/karpathy/neuraltalk2) by Andrej Karpathy. With OpenCV, it's easy to make this model caption live video or camera stream:

[![NeuralTalk2 Demo 1](https://cloud.githubusercontent.com/assets/9570420/14849852/69832384-0c86-11e6-9ef8-adfa0e7eba32.png)](http://cdn.makeagif.com/media/4-04-2016/eLgBBZ.gif)

[![NeuralTalk2 Demo 2](https://cloud.githubusercontent.com/assets/9570420/14849853/698439d6-0c86-11e6-9b75-fd9e5d8d17e1.png)](http://cdn.makeagif.com/media/4-04-2016/92PJ5o.gif)

[![NeuralTalk2 Demo 3](https://cloud.githubusercontent.com/assets/9570420/14849855/69963a96-0c86-11e6-92ff-723b143e99c7.png)](http://cdn.makeagif.com/media/4-04-2016/7ysYNO.gif)

The script [can be found](https://github.com/karpathy/neuraltalk2/blob/master/videocaptioning.lua) inside the NeuralTalk2 repository itself.

### Interactive Face Recognition with GPU

Another advantage of OpenCV is already mentioned NVIDIA CUDA support. It can help one to significantly speed up image processing and computer vision routines. These include image-processing-specific matrix operations, background segmentation, video [en/de]coding, feature detection and description, image filtering, object detection, computing optical flow, stereo correspondence etc.

Here is another code sample demonstrating some of the above features. This is an interactive face recognition application. After launched, it asks the user to manually classify people appearing in video stream into N (also user-defined) classes. When the number of labeled faces is sufficient for automatic recognition, discriminative descriptors are extracted using the convolutional neural network face descriptor and a SVM for classification is trained on them. Then the program switches to recognition mode and captions detected faces in stream with predicted "person name".

For speed, the face descriptor we use is the most lightweight (~3.7 millions of parameters) of [OpenFace](http://cmusatyalab.github.io/openface/) models, which are based on the CVPR 2015 paper [FaceNet: A Unified Embedding for Face Recognition](http://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf). It was pre-trained with a combination of [FaceScrub](http://vintage.winklerbros.net/facescrub.html) and [CASIA-WebFace](http://arxiv.org/abs/1411.7923) face recognition datasets.

![screenshot](https://cloud.githubusercontent.com/assets/9570420/13470424/2c5d3106-e0bd-11e5-9319-9f1dbf8c86ab.png)  
![screenshot 1](https://cloud.githubusercontent.com/assets/9570420/13470423/2c5d5064-e0bd-11e5-842c-d99157e22d6c.png)  
![screenshot 2](https://cloud.githubusercontent.com/assets/9570420/13530688/b1f694ac-e233-11e5-955c-df71688f472b.png)  
![screenshot 3](https://cloud.githubusercontent.com/assets/9570420/13530687/b1ceebd2-e233-11e5-8947-06684910aeff.png)

Let us introduce how OpenCV interface for Lua looks like in this case. As usual, there's a single `require` for every separate OpenCV package:

```lua
local cv = require 'cv'
require 'cv.highgui'       -- GUI: windows, mouse
require 'cv.videoio'       -- VideoCapture
require 'cv.imgproc'       -- resize, rectangle, putText
require 'cv.cudaobjdetect' -- CascadeClassifier
require 'cv.cudawarping'   -- resize
require 'cv.cudaimgproc'   -- cvtColor
cv.ml = require 'cv.ml'    -- SVM
```

The GUI commands are very high-level:

```lua
-- create two windows
cv.namedWindow{'Stream window'}
cv.namedWindow{ 'Faces window'}
cv.setWindowTitle{'Faces window', 'Grabbed faces'}
cv.moveWindow{'Stream window', 5, 5}
cv.moveWindow{'Faces window', 700, 100}

local function onMouse(event, x, y, flags)
   if event == cv.EVENT_LBUTTONDBLCLK then
      -- do something
   end
end

cv.setMouseCallback{'Stream window', onMouse}

cv.imshow{'Stream window', frame}
cv.imshow{'Faces window', gallery}
```

Here's how we set up the SVM:

```lua
-- SVM to classify descriptors in recognition phase
local svm = cv.ml.SVM{}
svm:setType         {cv.ml.SVM_C_SVC}
svm:setKernel       {cv.ml.SVM_LINEAR}
svm:setDegree       {1}
svm:setTermCriteria {{cv.TermCriteria_MAX_ITER, 100, 1e-6}}
```

...and use it:

```lua
-- svmDataX is a FloatTensor of size (#dataset x #features)
-- svmDataY is an IntTensor of labels of size (1 x #dataset)
svm:train{svmDataX, cv.ml.ROW_SAMPLE, svmDataY}

...

-- recognition phase
-- :predict() input is a FloatTensor of size (1 x #features)
local person = svm:predict{descriptor}
```

GPU calls and their CPU analogues look very much alike, with a couple of differences: first, they are placed in a separate table, and second, they handle `torch.CudaTensor`s.

```lua
-- convert to grayscale and store result in original image's blue (first) channel
cv.cuda.cvtColor{frameCUDA, frameCUDA:select(3,1), cv.COLOR_BGR2GRAY}

...

cv.cuda.resize{smallFaceCUDA, {netInputSize, netInputSize}, dst=netInputHWC}
```

Don't forget that these functions adapt to Cutorch stream and device settings, so calling `cutorch.setStream()`, `cutorch.streamWaitFor()`, `cutorch.setDevice()` etc. matters.

The whole runnable script is [available here](https://github.com/shrubb/torch-opencv-demos/blob/face-recognition/face_recognition/demo.lua).

### Live Image Stylization

The [Texture Networks: Feed-forward Synthesis of Textures and Stylized Images](http://arxiv.org/abs/1603.03417) paper proposes an architecture to stylize images with a feed-forward network, shipping with an [open source implementation in Torch](https://github.com/DmitryUlyanov/texture_nets/). It takes ~20 ms to process a single image with Tesla K40 GPU, and ~1000 ms with CPU. Having this, a tiny modification allows us to render any scene in a particular style in real time:

[![Demo 1](https://cloud.githubusercontent.com/assets/9570420/14849854/698c3b22-0c86-11e6-94ff-381a5cae1785.png)](http://i.makeagif.com/media/4-24-2016/0zb-UY.gif)

[![Demo 2](https://cloud.githubusercontent.com/assets/9570420/14849850/69828ec4-0c86-11e6-8609-bf3553450d9b.png)](http://i.makeagif.com/media/4-23-2016/Pk4ZAL.gif)

By the way, these very GIFs (originally in form of encoded videos) were rendered using OpenCV as well. There is a `VideoWriter` class that serves as a simple interface to video codecs. Here is a sketch of a program that encodes similar sequence of frames as a video file and saves it to disk:

```lua
local cv = require 'cv'
require 'cv.videoio' -- VideoWriter
require 'cv.imgproc' -- resize

local size = 256
local frameToSaveSize = {sz*2, sz}

local videoWriter = cv.VideoWriter{
   "sample.mp4",
   cv.VideoWriter.fourcc{'D', 'I', 'V', 'X'}, -- or any other codec
   fps = 25,
   frameSize = frameToSaveSize
}

if not videoWriter:isOpened() then
   print('Failed to initialize video writer. Possibly wrong codec or file name/path trouble')
   os.exit(-1)
end

for i = 1,numFrames do
   -- get next image; for example, read it from camera
   local frame = cv.resize{retrieveNextFrame(), {sz, sz}}

   -- the next frame in the resulting video
   local frameToSave = torch.Tensor(frameToSaveSize[2], frameToSaveSize[1], 3)

   -- first, copy the original frame into the left half of frameToSave:
   frameToSave:narrow(2, 1, sz):copy(frame)

   -- second, copy the processed (for example, rendered in painter style)
   -- frame into the other half:
   frameToSave:narrow(2, sz+1, sz):copy(someCoolProcessingFunction(frame))

   -- finally, tell videoWriter to push frameToSave into the video
   videoWriter:write{frameToSave}
end
```

[Here](https://github.com/szagoruyko/torch-opencv-demos/tree/master/texture_nets) goes the full code, including a model trained on *The Starry Night*. A version of this code is running at http://likemo.net

With these demos we just covered a little bit of what's possible to do with OpenCV+Torch7 and expect more awesome computer vision and deep learning applications and research tools to come from the community.

Acknowledgements
===
[Sergey Zagoruyko](https://github.com/szagoruyko) for putting up most of the demo code and creating sample screenshots.  
[Soumith Chintala](https://github.com/soumith) for support from the Torch side.<br>
[Dmitry Ulyanov](https://github.com/DmitryUlyanov) for providing demo and code for texture networks

The project is created and maintained by a [VisionLabs](http://visionlabs.ru/) team. We thank everyone who contributes to the project by making PRs and helping catch bugs.
