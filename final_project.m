clear all 
close all
clc 
%% loading and pre processing the images 

% to get the data set from my google drive 
%https://drive.google.com/file/d/179BCM227sEgoj2sQ23valJh1iXoQquOc/view?usp=sharing


% this data can be found in  https://www.iuii.ua.es/datasets/masati/
% the owrk inspired by the classification done in https://www.mdpi.com/2072-4292/10/4/511/htm
% the first classification problem will be to use neural network to
% classify coast vs sea data (both with no ships on it 
% load the data  
%files_name=["coast/c","water/w","land/l","detail/d","coast_ship/x","multi/m","multi/y","ship/s"];
files_name=["MASATI-v2/detail","MASATI-v2/coast_ship","MASATI-v2/multi","MASATI-v2/ship"];
foldernames=["MASATI-v2/coast_ship_labels","MASATI-v2/multi_labels","MASATI-v2/ship_labels"];
% storing the images in a data store 
imds = imageDatastore(files_name,"LabelSource","foldernames");
%  feature extraction 
% the hue ,saturation and value channels 
% I just need 4 samples from the detail photos as this is the photos
% without label 
n=1600;
y = randsample(n,4);
figure

% sample to demonstrate the idea of HSV colors
subplot(2,2,1)
RGB=imds.readimage(1);
HSV=rgb2hsv(RGB);
imagesc(RGB)
title(imds.Labels(y(1)))

% the Hue channel
subplot(2,2,2)
Hue_1=HSV(:,:,1);
histogram(Hue_1)
title('the Hue channel')

% the saturation channel
subplot(2,2,3)
Saturation_1=HSV(:,:,2);
histogram(Saturation_1)
title('the saturation channel')

% the value channel
subplot(2,2,4)
Value_1=HSV(:,:,3);
histogram(Value_1)
title('the value channel')
figure
subplot(2,2,1)
RGB=imds.readimage(10);
HSV=rgb2hsv(RGB);
imagesc(RGB)
title(imds.Labels(y(1)))

% the Hue channel
subplot(2,2,2)
Hue_1=HSV(:,:,1);
histogram(Hue_1)
title('the Hue channel')

% the saturation channel
subplot(2,2,3)
Saturation_1=HSV(:,:,2);
histogram(Saturation_1)
title('the saturation channel')

% the value channel
subplot(2,2,4)
Value_1=HSV(:,:,3);
histogram(Value_1)
title('the value channel')

%% add the bounding box label to the data 
% most of the documentries are using yolo to train such a model such as the work in this blog 
clear bdl_box_labels
% https://blogs.mathworks.com/deep-learning/2022/02/16/detection-of-ships-on-satellite-images-using-yolo-v2-model/
% the data set I am using is different and my approach is different 
% so we need bounding box labeled data (this labels is already given with
% the model
files_name=["coast_ship_labels/x","multi_labels/m","ship_labels/s"];
labels_name=unique(imds.Labels);
labels_name(2)=[];
num_of_datas=[1037,313,1027];
sz=[length(imds.Labels) 1];
varTypes={'cell'};
varNames={'ship'};
bdl_box_labels=table('Size',sz,'VariableTypes',varTypes,'VariableNames',varNames);
for i=1:length(files_name)
    file_name=files_name(i);
    label=labels_name(i);
    num_of_data=num_of_datas(i);
    bdl_box=create_bdl_box_table(file_name,num_of_data);
    IDX_label=find(strcmp(string(imds.Labels),string(label)));
    bdl_box_labels.ship(IDX_label)=table2array(bdl_box);
    clear bdl_box
end

%%  creating bounding box for the one that doesnt have 

IDX_detail=find(strcmp(string(imds.Labels),"detail"));
photos_label=imds.Files(IDX_detail);
j=0;
for i =1:length(IDX_detail)
    photo=photos_label(i);
    photo=rgb2hsv(imread(photo{1}));
    value_min=0.45;
    levels=[0,1;0,1;value_min,1];  % the first row hue min,max and the second saturation ,the third value
    maskedRGBImage=mask_HSVlevels(photo,levels);
    % get the connected area
    CC = bwconncomp(maskedRGBImage);
    % apply lower masking value to detect lower brightness 
    while CC.NumObjects ==0
        value_min=value_min-0.2;
        levels=[0,1;0,1;value_min,1];
        maskedRGBImage=mask_HSVlevels(photo,levels);
        CC = bwconncomp(maskedRGBImage);
    end
    % if more than one object found remove the smal objects 
    numPixels = cellfun(@numel,CC.PixelIdxList);
    IDX=find(numPixels ~= max(numPixels));
    CC.PixelIdxList(IDX)=[];
    CC.NumObjects=1;
    box=regionprops(CC);
    box=box.BoundingBox(1,[1,2,4,5]);
    bbMatrix = vertcat(box);
    if box(1)==.5 || box(2)==.5
        j=j+1;
        indexes(j)=i;
        box=[200,200,60,80];
    end
    bdl_box_labels.ship(IDX_detail(i))={box}; 
end 

%%  preprocesss the photos for the deep learning model
% some photos are 1 channel while majority is 3 channels so this part to
% remove it 
j=0;
k=0;
for i=1:size(imds.Files)
    img = readimage(imds,i);
    
    if length(size(img))==length([512,512,3])
        j=j+1;
        idx1(j)=i;
    else
        k=k+1;
        idx(k)=i;
    end
end
%%
imdsnew=subset(imds,idx1);
bdl_box_labels(idx,:)=[];
%%  check the clusters of anchor boxes / this part is from the documentation 
% the idea is to cluster some boxes together 
% then use the estimate anchor box function to find this boxes 
allBoxes = vertcat(bdl_box_labels.ship{:});

aspectRatio = allBoxes(:,3) ./ allBoxes(:,4);
area = prod(allBoxes(:,3:4),2);

figure
scatter(area,aspectRatio)
xlabel("Box Area")
ylabel("Aspect Ratio (width/height)");
title("Box Area vs. Aspect Ratio")

%%  splitting data into training and testing data with 85% training will have some validation data 
perc_of_test_data=.15;
bdl_box_labels=table2array(bdl_box_labels);
[Train_val_data,Test_data,Train_val_label,Test_label]=split_data(imdsnew.Files,bdl_box_labels,perc_of_test_data);
% now split the data into some training and validation data 
[Train_data,val_data,Train_label,val_label]=split_data(Train_val_data,Train_val_label,perc_of_test_data);
% now we need to create a box label data store 
%% pre process the data resizing the image and the boxlabeldatastore 
% the yolo toolbox deal with image datastores so we will create image and
% labels data stores 

ship=Train_label;
training_label_table=cell2table(ship);
bldsTrain = boxLabelDatastore(training_label_table);

training_data_table=cell2table(Train_data);
imdsTrain = imageDatastore(training_data_table{:,'Train_data'});
% val data 

ship=val_label;
val_label_table=cell2table(ship);
bldsVal = boxLabelDatastore(val_label_table);

val_data_table=cell2table(val_data);
imdsVal = imageDatastore(val_data_table{:,'val_data'});
% test data

ship=val_label;
test_label_table=cell2table(ship);
bldsTest = boxLabelDatastore(test_label_table);

test_data_table=cell2table(Test_data);
imdsTest = imageDatastore(test_data_table{:,'Test_data'});

%%% now build the data with combine to combine box labels and photos 
trainingData = combine(imdsTrain,bldsTrain);
validationData = combine(imdsVal,bldsVal);
testData = combine(imdsTest,bldsTest);

% resizing the data using the 
inputSize=[224 224 3];     % to the network 
numClasses=1;
scale=224/512;
% resize trainning data
augmentedTrainingData = transform(trainingData,@augmentData);
preprocessedTrainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
% resize validation data
augmentedValidationData = transform(validationData,@augmentData);
preprocessedValidationData = transform(augmentedValidationData,@(data)preprocessData(data,inputSize));

%%  the yolo model 

% estimate anchor box for classifiere
numAnchors = 7;

[anchorBoxes, meanIoU] = estimateAnchorBoxes(preprocessedTrainingData, numAnchors);
% we will use resnet50 for feauture extraction 
% the resnet 50 is faster than the resnet 101 and it consists of lower
% layer but if it can give high accuraccy so be it 

%network=resnet50();
network = resnet101();
%network=densenet201();
 
%%
% the feature layer activation_40_relu is with resnet 50

%featureLayer='activation_40_relu';

% the feature layer 'res4b18_relu' is with resnet101
featureLayer = 'res4b18_relu';
% the feature layer 'conv5_block30_0_relu' is with densenet201
%featureLayer ='conv5_block30_0_relu';


%%% create the object detection network 

lgraph = yolov2Layers(inputSize,numClasses,anchorBoxes,network,featureLayer);

options = trainingOptions('adam', ...
        'MiniBatchSize',16, ....
        'InitialLearnRate',1e-3, ...
        'LearnRateDropFactor',0.2,...
        'LearnRateDropPeriod',5, ...
        'MaxEpochs',10, ... 
        'CheckpointPath',tempdir...
        ,'ValidationData',preprocessedValidationData,...
        'Plots','training-progress');
%%  

if isfile ('ship_detection_neural_networkresnet101.mat')       
    % load the Yolo netwrok 
    load('ship_detection_neural_networkresnet101.mat');
else
    % Train the network
    [detector,info] = trainYOLOv2ObjectDetector(preprocessedTrainingData,lgraph,options);
    % to save the trained network 
    ship_detection_neural_network=detector;
    save ship_detection_neural_networkresnet101
end

% if isfile ('ship_detection_neural_networkresnet50.mat')       
%     % load the Yolo netwrok 
%     load('ship_detection_neural_networkresnet50.mat');
%     
% else
%     % Train the network
%     [detector,info] = trainYOLOv2ObjectDetector(preprocessedTrainingData,lgraph,options);
%     % to save the trained network 
%     ship_detection_neural_network=detector;
%     save ship_detection_neural_networkresnet50
% end

%% show results 
j=0;
k=0;
figure(3)
z=0;
y=0;
for i =1:length(Test_data)
   
    I = imread(Test_data{i});
    I = imresize(I,inputSize(1:2));
    %imshow(I)
    %figure 
%detector=pretrained.detector;
   [bboxes,scores] = detect(detector,I);
   [predicted_number_of_boxes,~]=size(bboxes);
   [true_number_of_boxes,~]=size(cell2mat(Test_label(i)));
   if isempty(bboxes)==true
       j=j+1;
       FN(j)=i;
   elseif predicted_number_of_boxes==true_number_of_boxes 
       y=y+1;
       TP(y)=i;
       %this part is to get the plotting to show the results 
       if  k<4 && predicted_number_of_boxes==1
       k=k+1;
       I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
       subplot(2,2,k)
       imshow(I)
       title("Accuracy of detecting the object=  "+string(scores*100)+ " %")
       elseif z<2 && predicted_number_of_boxes>1 && predicted_number_of_boxes==true_number_of_boxes
           z=z+1;
       elseif  z==2 &&  predicted_number_of_boxes>1 && predicted_number_of_boxes==true_number_of_boxes
       figure
       z=z+1;
       I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
       imshow(I)
       %title("Accuracy of detecting the object=  "+string(scores*100)+ " %")
       end 
   end
end
%%
FP=(length(Test_data)-j)-y ;
precision=length(TP)/(length(TP)+FP) ;    % FP found an extra object or define non ship as a ship 
recall=length(TP)/(length(TP)+length(FN));  %( FN is not finding the object which should be found )
F1_Score=(precision*recall)/((precision+recall)/2);


%% functions definitions 
%%% getting the ship label(x,y,w,h)  from the data label . this describe the box or the area where the ship is located to create a bounding box to train the data 

function bdl_box_table=create_bdl_box_table(file_name,num_of_data)

%   this function is to create bounding box tables from the given bounding
%   box labeled data given in the labeled data folders 
     ship={};
     bdl_box_table=table(ship);
     indexes=[];
     for i =1:num_of_data
         % this part is to deal with the format of the file and  if any
         % file is missing 
          ship=[];
          if i<10 && isfile("MASATI-v2/"+file_name+"000"+string(i)+".xml") % to check if the file exist

               file=readstruct("MASATI-v2/"+file_name+"000"+string(i)+".xml");

          elseif i<100 && isfile("MASATI-v2/"+file_name+"00"+string(i)+".xml")

               file=readstruct("MASATI-v2/"+file_name+"00"+string(i)+".xml");

          elseif i<1000 && isfile("MASATI-v2/"+file_name+"0"+string(i)+".xml")

               file=readstruct("MASATI-v2/"+file_name+"0"+string(i)+".xml");

          elseif isfile("MASATI-v2/"+file_name+string(i)+".xml")

               file=readstruct ("MASATI-v2/"+file_name+string(i)+".xml");
          else 
               indexes=[indexes,i];
               continue
          end
          if length(file.object)==1
               xmin=file.object.bndbox.xmin;
               xmax=file.object.bndbox.xmax;
               ymin=file.object.bndbox.ymin;
               ymax=file.object.bndbox.ymax;
               ship={[xmin,ymin,xmax-xmin,ymax-ymin]};
          else 
                for j=1:length(file.object)
                    xmin=file.object(j).bndbox.xmin;
                    xmax=file.object(j).bndbox.xmax;
                    ymin=file.object(j).bndbox.ymin;
                    ymax=file.object(j).bndbox.ymax;
                    ship=[ship;xmin,ymin,xmax-xmin,ymax-ymin];
                 end
                 ship={ship};
           end
           bdl_box_table.ship(i)=ship;  
     end
     bdl_box_table(indexes,:)=[];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%% splitting data into training and testing sets
function [Train_Data,Test_Data,Train_label,Test_label]=split_data(DS,labels,perc_of_test_data)
    n=length(DS);
    hpartition = cvpartition(n,'Holdout',perc_of_test_data);  % create partition with percentage 
    idxTrain = training(hpartition);  % training indexes
    idxTest= test(hpartition);        % test indexes
    Train_Data=DS(idxTrain);        % Train data
    Test_Data=DS(idxTest);  
    Train_label=labels(idxTrain,:);        % Train data
    Test_label=labels(idxTest,:);
end

function maskedRGBImage=mask_HSVlevels(photo,levels)
%%%% levels [ HueMin,HueMax;
%            saturationMin, saturationMax;
%            ValueMin, ValueMax]
% maskin the area 
    HueMin = levels(1,1);
    HueMax = levels(1,2);
% Define thresholds for 'Saturation'
    saturatMin = levels(2,1);
    saturatioMax = levels(2,2);
% Define thresholds for 'Value'
    ValueMin = levels(3,1);
    ValueMax = levels(3,2);
% Create mask based on chosen histogram thresholds
    BW = ( (photo(:,:,1) >= HueMin) | (photo(:,:,1) <= HueMax) ) & ...
         (photo(:,:,2) >= saturatMin ) & (photo(:,:,2) <= saturatioMax) & ...
         (photo(:,:,3) >= ValueMin ) & (photo(:,:,3) <= ValueMax);
% Initialize output masked image based on input image.
     maskedRGBImage = photo;
% Set background pixels where BW is false to zero.
     maskedRGBImage(repmat(~BW,[1 1 3])) = 0;
end

%%%%%%%%%%%%%% matlab functions to preprocess the data with yolo with some
%%%%%%%%%%%%%% modification 
function B = augmentData(A)
% Apply random horizontal flipping, and random X/Y scaling. Boxes that get
% scaled outside the bounds are clipped if the overlap is above 0.25. Also,
% jitter image color.
B = cell(size(A));
I = A{1};
sz = size(I);
if numel(sz)==3 && sz(3) == 3
    I = jitterColorHSV(I,...
        'Contrast',0.2,...
        'Hue',0,...
        'Saturation',0.1,...
        'Brightness',0.2);
end
% Randomly flip and scale image.
tform = randomAffine2d('XReflection',true,'Scale',[1 1.1]);
rout = affineOutputView(sz,tform,'BoundsStyle','CenterOutput');
B{1} = imwarp(I,tform,'OutputView',rout);
% Apply same transform to boxes.
boxEstimate=round(A{2});
boxEstimate(:,1)=max(boxEstimate(:,1),1);
boxEstimate(:,2)=max(boxEstimate(:,2),1);
[B{2},indices] = bboxwarp(boxEstimate,tform,rout,'OverlapThreshold',0.25);
B{3} = A{3}(indices);
% Return original data only when all boxes are removed by warping.
if isempty(indices)
    B = A;
end
end
function data = preprocessData(data,targetSize)
scale = targetSize(1:2)./size(data{1},[1 2]);
data{1} = imresize(data{1},targetSize(1:2));
boxEstimate=round(data{2});
boxEstimate(:,1)=max(boxEstimate(:,1),1);
boxEstimate(:,2)=max(boxEstimate(:,2),1);
data{2} = bboxresize(boxEstimate,scale);
end
