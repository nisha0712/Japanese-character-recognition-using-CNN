
a = readNPY('C:\Users\deepa\Downloads\jap\k49-train-imgs\arr_0.npy');
b = readNPY('C:\Users\deepa\Downloads\jap\k49-train-labels\arr_0.npy');
c = readNPY('C:\Users\deepa\Downloads\jap\k49-test-imgs\arr_0.npy');
d = readNPY('C:\Users\deepa\Downloads\jap\k49-test-labels\arr_0.npy');

for i=1:size(a,1)
    train_image(:,:,1,i) = a(i,:,:)./255;
end

train_label = b';

for i=1:size(c,1)
    test_image(:,:,1,i) = c(i,:,:)./255;
end

test_label = d';

 
layers = [
    imageInputLayer([28 28 1],'Name','Input layer')
    
    convolution2dLayer(5,32,'Padding','same','Name','Conv_1')
    batchNormalizationLayer('Name','Norm_1')
    reluLayer('Name','relu_1')
    
    maxPooling2dLayer(2,'Stride',2,'Name','pool_1')
    
    convolution2dLayer(3,64,'Padding','same','Name','Conv_2')
    batchNormalizationLayer('Name','norm_2')
    reluLayer('Name','relu_2')
    
    maxPooling2dLayer(2,'Stride',2,'Name','pool_2')
    
    convolution2dLayer(3,128,'Padding','same','Name','conv_3')
    batchNormalizationLayer('Name','norm_3')
    reluLayer('Name','relu_3')
    
    fullyConnectedLayer(49,'Name','fully_connected')
    softmaxLayer('Name','soft_max')
    classificationLayer('Name','classify')];
 
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'Verbose',true, ...
    'Plots','training-progress');

net = trainNetwork(train_image,categorical(train_label),layers,options);

prediction = classify(net,test_image);

mat = confusionmat(categorical(test_label),prediction);
para = parameters_m(mat);

accuracy = sum(para(1,:));