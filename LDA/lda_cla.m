clear
clc
% load data
%   train_data  (1024,2387)
%   train_label (1,2387)
%   test_data   (1024,1023)
%   test_label  (1,1023)
load('../facedata.mat');

% LDA
mu = mean(train_data,2);
for i = 1:21
    if i <= 20
        train{i} = train_data(:,(i-1)*119+1:i*119);
        mu_i{i} = mean(train{i},2);
        S_i{i} = (train{i}-mu_i{i})*(train{i}-mu_i{i})'/119;
    else
        train{i} = train_data(:,2381:2387);
        mu_i{i} = mean(train{i},2);
        S_i{i} = (train{i}-mu_i{i})*(train{i}-mu_i{i})'/7;
    end
end
S_W = zeros(1024,1024);
S_B = zeros(1024,1024);
for i = 1:21
    if i <= 20
        S_W = S_W + S_i{i}*119/2387;
        S_B = S_B + (mu_i{i}-mu)*(mu_i{i}-mu)'*119/2387;
    else
        S_W = S_W + S_i{i}*7/2387;
        S_B = S_B + (mu_i{i}-mu)*(mu_i{i}-mu)'*7/2387;
    end
end
[W,Lam] = eig(S_B,S_W);

k = 5; % k-nearest neighbor

train_2 = W(:,1:2)'*train_data; % 2x2387
test_2 = W(:,1:2)'*test_data; % 2x1023
X1_square = repmat(sum(test_2.^2,1),2387,1)'; %1023x2387
X2_square = repmat(sum(train_2.^2,1),1023,1); %1023x2387
A = 2*test_2'*train_2;
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
fprintf('dimansionality 2: PIE:%.2f%% Mine:%.2f%% \n',acc_pie*100,acc_mine*100);

train_3 = W(:,1:3)'*train_data; % 2x2387
test_3 = W(:,1:3)'*test_data; % 2x1023
X1_square = repmat(sum(test_3.^2,1),2387,1)'; %1023x2387
X2_square = repmat(sum(train_3.^2,1),1023,1); %1023x2387
A = 2*test_3'*train_3;
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
fprintf('dimansionality 3: PIE:%.2f%% Mine:%.2f%% \n',acc_pie*100,acc_mine*100);


train_9 = W(:,1:9)'*train_data; % 2x2387
test_9 = W(:,1:9)'*test_data; % 2x1023
X1_square = repmat(sum(test_9.^2,1),2387,1)'; %1023x2387
X2_square = repmat(sum(train_9.^2,1),1023,1); %1023x2387
A = 2*test_9'*train_9;
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
fprintf('dimansionality 9: PIE:%.2f%% Mine:%.2f%% \n',acc_pie*100,acc_mine*100);


train_40 = W(:,1:40)'*train_data; % 2x2387
test_40 = W(:,1:40)'*test_data; % 2x1023
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




