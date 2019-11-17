clear
clc
% load data
%   train_data  (1024,2387)
%   train_label (1,2387)
%   test_data   (1024,1023)
%   test_label  (1,1023)
load('../facedata.mat');


% PCA
%   centralize the data
train_mean = mean(train_data,2);
train_cen = train_data - repmat(train_mean,1,2387);
%   svd
[U,S,V] = svd(train_cen);
lam = S*S';

train_data_2d = U(:,1:2)'*train_data;
train_data_3d = U(:,1:3)'*train_data;
rng(666) % control random seed
rand_idx_train = randperm(2380,498);
rand_idx_mine = 2380 + randperm(7,2);

% 2d visualization
figure()
hold on
s1 = scatter(train_data_2d(1,rand_idx_train),train_data_2d(2,rand_idx_train),10,train_label(1,rand_idx_train),'filled');
s2 = scatter(train_data_2d(1,rand_idx_mine),train_data_2d(2,rand_idx_mine),36,'r','pentagram','filled');
grid on
title('PCA 2D visualization ')
legend([s1 s2],{'PIE','MINE'})
hold off

% 3d visualization
figure()
hold on
s3 = scatter3(train_data_3d(1,rand_idx_train),train_data_3d(2,rand_idx_train),train_data_3d(3,rand_idx_train),10,train_label(1,rand_idx_train),'filled');
s4 = scatter3(train_data_3d(1,rand_idx_mine),train_data_3d(2,rand_idx_mine),train_data_3d(3,rand_idx_mine),36,'r','pentagram','filled');
grid on
title('PCA 3D visualization ')
legend([s3 s4],{'PIE','MINE'})
view([1 2 1])
hold off

% visualize eigenfaces
for i = 1:3
    eigen_face = reshape(U(:,i),32,32);
    eigen_face = (eigen_face - min(eigen_face,[],'all'))./max(eigen_face,[],'all');
    figure()
    imshow(eigen_face);
    truesize([200 200]);
    title(sprintf('Eigenface%d',i))
    imwrite(eigen_face,sprintf('eigen_face_%d.png',i))
end
