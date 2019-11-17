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

train_data_2d = W(:,1:2)'*train_data;
train_data_3d = W(:,1:3)'*train_data;
test_data_2d = W(:,1:2)'*test_data;
test_data_3d = W(:,1:3)'*test_data;
rng(666) % control random seed
rand_idx_train = randperm(2380,498);
rand_idx_mine = 2380 + randperm(7,2);

% 2d visualization
figure()
hold on
s1 = scatter(train_data_2d(1,rand_idx_train),train_data_2d(2,rand_idx_train),10,train_label(1,rand_idx_train),'filled');
s2 = scatter(train_data_2d(1,rand_idx_mine),train_data_2d(2,rand_idx_mine),36,'r','pentagram','filled');
s3 = scatter(test_data_2d(1,1021:1023),test_data_2d(2,1021:1023),36,'black','pentagram','filled');
grid on
title('LDA 2D visualization ')
legend([s1 s2 s3],{'PIE','MINE', 'test-MINE'},'Location','northwest')
hold off

% 3d visualization
figure()
hold on
s3 = scatter3(train_data_3d(1,rand_idx_train),train_data_3d(2,rand_idx_train),train_data_3d(3,rand_idx_train),10,train_label(1,rand_idx_train),'filled');
s4 = scatter3(train_data_3d(1,rand_idx_mine),train_data_3d(2,rand_idx_mine),train_data_3d(3,rand_idx_mine),36,'r','pentagram','filled');
s5 = scatter3(test_data_3d(1,1021:1023),test_data_3d(2,1021:1023),test_data_3d(3,1021:1023),36,'black','pentagram','filled');
grid on
title('LDA 3D visualization ')
legend([s3 s4 s5],{'PIE','MINE','test-MINE'})
view([1 2 1])
hold off







