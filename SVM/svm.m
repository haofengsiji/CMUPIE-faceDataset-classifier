clear
clc
% load data
%   train_data  (1024,2387)
%   train_label (1,2387)
%   test_data   (1024,1023)
%   test_label  (1,1023)
load('../facedata.mat');

% load libSVM
addpath('libsvm-3.24\matlab')

% PCA
% centralize the data
train_mean = mean(train_data,2);
train_cen = train_data - train_mean;
% svd
[U,S,V] = svd(train_cen);
lam = S*S';

train_80 = U(:,1:80)'*train_data; % 80x2387
test_80 = U(:,1:80)'*test_data; % 80x1023
model_11 = svmtrain(train_label', train_80', '-t 0 -c 2');
model_1 = svmtrain(train_label', train_80', '-t 0 -c 1');
model_2 = svmtrain(train_label', train_80', '-t 0 -c 0.1');
model_3 = svmtrain(train_label', train_80', '-t 0 -c 0.01');

[~, accuracy_11, ~] = svmpredict(test_label', test_80', model_11); 
[~, accuracy_1, ~] = svmpredict(test_label', test_80', model_1); 
[~, accuracy_2, ~] = svmpredict(test_label', test_80', model_2); 
[~, accuracy_3, ~] = svmpredict(test_label', test_80', model_3); 

train_200 = U(:,1:200)'*train_data; % 200x2387
test_200 = U(:,1:200)'*test_data; % 200x1023
model_44 = svmtrain(train_label', train_200', '-t 0 -c 2');
model_4 = svmtrain(train_label', train_200', '-t 0 -c 1');
model_5 = svmtrain(train_label', train_200', '-t 0 -c 0.1');
model_6 = svmtrain(train_label', train_200', '-t 0 -c 0.01');

[~, accuracy_44, ~] = svmpredict(test_label', test_200', model_44); 
[~, accuracy_4, ~] = svmpredict(test_label', test_200', model_4); 
[~, accuracy_5, ~] = svmpredict(test_label', test_200', model_5); 
[~, accuracy_6, ~] = svmpredict(test_label', test_200', model_6); 
