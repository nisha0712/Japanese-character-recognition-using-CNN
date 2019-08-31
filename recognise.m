n1 = im2uint8(n1);

n1 = imresize(n1,[28 28]);
n = reshape(n1, [28 28 1]);

z = classify(net,n);

% layers = [
%     imageInputLayer([28 28 1],'Name','Input layer')
%     
%     convolution2dLayer(5,32,'Padding','same','Name','Conv_1')
%     batchNormalizationLayer('Name','Norm_1')
%     reluLayer('Name','relu_1')
%     
%     maxPooling2dLayer(2,'Stride',2,'Name','pool_1')
%     
%     convolution2dLayer(3,64,'Padding','same','Name','Conv_2')
%     batchNormalizationLayer('Name','norm_2')
%     reluLayer('Name','relu_2')
%     
%     maxPooling2dLayer(2,'Stride',2,'Name','pool_2')
%     
%     convolution2dLayer(3,128,'Padding','same','Name','conv_3')
%     batchNormalizationLayer('Name','norm_3')
%     reluLayer('Name','relu_3')
%     
%     fullyConnectedLayer(49,'Name','fully_connected')
%     softmaxLayer('Name','soft_max')
%     classificationLayer('Name','classify')];
% lgraph = layerGraph(layers);
% plot(lgraph);