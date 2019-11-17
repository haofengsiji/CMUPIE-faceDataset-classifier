%dataloader split CMU PIE 1~20 subjects and personal subjects into train
%set(70%) and test set(30%) and save data in facedata.mat
%   train_data  (1024,2387)
%   train_label (1,2387)
%   test_data   (1024,1023)
%   test_label  (1,1023)
train_data = zeros(1024,2387);
train_label = zeros(1,2387);
test_data = zeros(1024,1023);
test_label = zeros(1,1023);
train_count = 1;
test_count = 1;

rng(666) % control random seed

for i = 1:21
    path = sprintf('FACE/%d', i);
    dir_ls = dir(fullfile(path,'*.jpg'));
    num = numel(dir_ls);
    idx = 1:num;
    shuffle_idx = idx(randperm(num));
    for j = 1:num
        img = double(imread(fullfile(path,dir_ls(shuffle_idx(j)).name)))/255;
%         img = double(imread(fullfile(path,dir_ls(j).name)))/255;
        if j <= int32(num*0.7)
            train_data(:,train_count) = reshape(img,1024,1);
            train_label(:,train_count) = i;
            train_count = train_count+1;
        else
            test_data(:,test_count) = reshape(img,1024,1);
            test_label(:,test_count) = i;
            test_count = test_count+1;
        end
    end     
end
save('facedata.mat','train_data','train_label','test_data','test_label');