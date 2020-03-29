function direcs = two_nets_predict_direcs(images)
% This function receives as input images and returns the body angle for each image.
% First it will predict all images using the MainNet, a net that works on 50X50. 
% We will smooth this angle vector with a 7 sized window.
% Afterwards it will calculate the variance over 100 elements of the first derivative of the angles vector.
% For every value of this parameter that will be larger than the threshold(40),
% we will predict again those 100(VAR_CONST) frames with the HardNet,
% a net that works on 100X100 images and therefore is slower but more accurate.
% We will also smooth this 100 frame vector with an 11 sized window. 
% We will insert these 100 vectors instead of the prediction of the MainNet.
% In order to keep the continuity, when inserting this vector we will use a
% ramp for the window- average between the first 10 (WINDOW_RAMP_SIZE)
% values of the HardNet and the MainNet.
% 
% input:
%   images: 100x100xN(or 200x200xN) vector of images
%output:
%   direcs: Nx1 directions vector of the images in the range of 1:360

THRESHOLD = 40; %threshold of the hard parts of the variance derivative.
WINDOW_RAMP_SIZE = 10; %the size of the part where we merge the two net- for example if the window size is 120 than it will be 10 - for the start and the end of the window
VAR_CONST = 100; %number of element to do variance on, for var_deriv;
%#### check net size ####
if size(images , 1) == 200
    images = images(1:2:200,1:2:200,:);
elseif size(images , 1) ~= 100
    disp('Wrong image size! Please enter 200X200 or 100x100 images.')
end
n_images = size(images,3);

% load the nets and predict using only main net.
hard_net = load('net1_hardNet.mat', 'net');
main_net =  load('net50_mainNet.mat', 'net');
main_preds = predict_direcs(images(1:2:end,1:2:end,:), {main_net(1).net}, true); %predict with aug
main_preds = smooth_angles(main_preds, 7);%smoothing before gettig the var_deriv

% #### load insted of predict for testing ####
%main_preds = load('net50_preds_unsmoothed.mat','val_direcs_310119');
%main_preds = smooth_angles(main_preds.val_direcs_310119, 7);%smoothing before gettig the var_deriv
% #### turnoff after finish testing ####

%find the hard parts, where our main net fails, by allocating where the variance of the derivartive
%is greater than the threshold
var_deriv = variance_derivative(main_preds, VAR_CONST); %create a mooving variance on each 100 elements of the derivative of the preds
hard_parts_idx =  find(var_deriv > THRESHOLD);
hard_parts_idx = hard_parts_idx*VAR_CONST - (VAR_CONST-1); %getting the exact indexing

%create mixed_vector of preds from main and hard
mixed_vec = main_preds;
size_hard_parts_idx = length(hard_parts_idx);

% The windows are used for keeping the continuity, when inserting the
% HardNet vector. We will average between the first 10 values of the HardNet and the MainNet.
%windows is a cell array of windows - 1 is without ramps, 2 is with ram up
%3 is with ramp down, 4 is with ramps up and down.
window_2 = create_window(VAR_CONST,WINDOW_RAMP_SIZE,0); % ramp up
window_3 = create_window(VAR_CONST,0,WINDOW_RAMP_SIZE); %ramp down
window_4 = create_window(VAR_CONST,WINDOW_RAMP_SIZE,WINDOW_RAMP_SIZE); %ramp up & down
windows = {ones(1,VAR_CONST),window_2, window_3, window_4};
for i = 1: size_hard_parts_idx
    %find which window we need and what is the indexes we wish to use the hard_net
    [window_idx, start_idx, end_idx]= get_window_idx(i, hard_parts_idx,VAR_CONST,...
    n_images, WINDOW_RAMP_SIZE);
    window = windows{window_idx};
    %predict the directions of the hard images with the hard_net
    end_idx = (n_images > end_idx)*end_idx + (n_images < end_idx)*n_images; %make sure end_idx <= n_images
    %In order to make smoothing of the hard net correctly on the edges, we will predict 5 more images
    %on each edege without ramp, and than smooth and croop. Only if were are not out of bound
    if (window_idx ==1)&&(start_idx ~= 1) && (end_idx + 5 < n_images) %Do the smoothing manipulation
        hard_net_preds = predict_direcs(images(:,:,start_idx-5:end_idx+5), {hard_net(1).net}, true); 
        hard_net_preds =  smooth_angles(hard_net_preds, 11); %smooth hard preds with 11
        hard_net_preds = hard_net_preds(6:105); 
    elseif (window_idx == 2)&&(end_idx + 5 < n_images)
        hard_net_preds = predict_direcs(images(:,:,start_idx:end_idx+5), {hard_net(1).net}, true); 
        hard_net_preds =  smooth_angles(hard_net_preds, 11); %smooth hard preds with 11
        hard_net_preds = hard_net_preds(1:end - 5); 
    elseif (window_idx == 3) &&(start_idx ~= 1)
        hard_net_preds = predict_direcs(images(:,:,start_idx-5:end_idx), {hard_net(1).net}, true); 
        hard_net_preds =  smooth_angles(hard_net_preds, 11); %smooth hard preds with 11
        hard_net_preds = hard_net_preds(6:end); 
    else
       hard_net_preds = predict_direcs(images(:,:,start_idx:end_idx), {hard_net(1).net}, true); 
       hard_net_preds =  smooth_angles(hard_net_preds, 11); %smooth hard preds with 11
    end
    %set the hard_net preds in the mixed vector
    mixed_vec(start_idx : end_idx) = main_preds(start_idx:end_idx).*transpose(1-window)...
        + hard_net_preds.*transpose(window);
end
direcs = mixed_vec;
end

