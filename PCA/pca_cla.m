clear
clc
% load data
%   train_data  (1024,2387)
%   train_label (1,2387)
%   test_data   (1024,1023)
%   test_label  (1,1023)
load('../facedata.mat');

% centralize the data
train_mean = mean(train_data,2);
train_cen = train_data - train_mean;
% svd
[U,S,V] = svd(train_cen);
lam = S*S';

k = 1; % k-nearest neighbor

train_40 = U(:,1:40)'*train_data; % 40x2387
test_40 = U(:,1:40)'*test_data; % 40x1023
X1_square = repmat(sum(test_40.^2,1),2387,1)'; %1023x2387
X2_square = repmat(sum(train_40.^2,1),1023,1); %1023x2387
A = 2*test_40'*train_40;
dists_l2 = X1_square + X2_square - A;

pred = zeros(1,1023);
for i = 1:1023
    [~,idx] = sort(dists_l2(i,:));
    closest_label = train_label(:,idx(1:k));
    [~, argmax] = max(histcounts(closest_label,[1:22]));
    pred(:,i) = argmax;
end

acc_pie = sum(pred(:,1:1020)==test_label(:,1:1020),'all')/1020;
acc_mine = sum(pred(:,1021:1023)==test_label(:,1021:1023),'all')/3;
fprintf('dimansionality 40: PIE:%.2f%% Mine:%.2f%% \n',acc_pie*100,acc_mine*100);

train_80 = U(:,1:80)'*train_data; % 80x2387
test_80 = U(:,1:80)'*test_data; % 80x1023
X1_square = repmat(sum(test_80.^2,1),2387,1)'; %1023x2387
X2_square = repmat(sum(train_80.^2,1),1023,1); %1023x2387
A = 2*test_80'*train_80;
dists_l2 = X1_square + X2_square - A;

pred = zeros(1,1023);
for i = 1:1023
    [~,idx] = sort(dists_l2(i,:));
    closest_label = train_label(:,idx(1:k));
    [~, argmax] = max(histcounts(closest_label,[1:22]));
    pred(:,i) = argmax;
end

acc_pie = sum(pred(:,1:1020)==test_label(:,1:1020),'all')/1020;
acc_mine = sum(pred(:,1021:1023)==test_label(:,1021:1023),'all')/3;
fprintf('dimansionality 80: PIE:%.2f%% Mine:%.2f%% \n',acc_pie*100,acc_mine*100);

train_200 = U(:,1:200)'*train_data; % 200x2387
test_200 = U(:,1:200)'*test_data; % 200x1023
X1_square = repmat(sum(test_200.^2,1),2387,1)'; %1023x2387
X2_square = repmat(sum(train_200.^2,1),1023,1); %1023x2387
A = 2*test_200'*train_200;
dists_l2 = X1_square + X2_square - A;

pred = zeros(1,1023);
for i = 1:1023
    [~,idx] = sort(dists_l2(i,:));
    closest_label = train_label(:,idx(1:k));
    [~, argmax] = max(histcounts(closest_label,[1:22]));
    pred(:,i) = argmax;
end

acc_pie = sum(pred(:,1:1020)==test_label(:,1:1020),'all')/1020;
acc_mine = sum(pred(:,1021:1023)==test_label(:,1021:1023),'all')/3;
fprintf('dimansionality 200: PIE:%.2f%% Mine:%.2f%% \n',acc_pie*100,acc_mine*100);
