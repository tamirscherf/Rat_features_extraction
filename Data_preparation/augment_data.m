function augment_data()
    % Load a image file, augment it by the enlargment factor and  save it. 
    [data,filename] = load_data();
    IMG_SIZE = 200;
    K = 64; %enlargement factor
    numOfImages = size(data.('image'),3);
    augSize = numOfImages*K;
    
    %intilaizing matrixs for the augmented data
    images = zeros(IMG_SIZE, IMG_SIZE, augSize, 'uint8');
    direcs_body = zeros(augSize, 1);
    direcs_head = zeros(augSize, 1);
    locs_head = zeros(augSize, 2);
    locs_neck = zeros(augSize, 2);
    locs_base = zeros(augSize, 2);
    difficulty_level = zeros(augSize, 1);
    
    %augment each image for loop
    for i = 0:(numOfImages-1) 
        imgAugmented = augment_one_image(data, i+1, K);
        %insert data of augmenting one image to the matrixs
        images(:, :, (i*K +1):(i+1)*K) = imgAugmented.('image');
        direcs_body((i*K +1):(i+1)*K, 1) = imgAugmented.('bodyAngle');
        direcs_head((i*K +1):(i+1)*K, 1) = imgAugmented.('headAngle');
        locs_head((i*K +1):(i+1)*K, :) = imgAugmented.('headPoint');
        locs_neck((i*K +1):(i+1)*K, :) = imgAugmented.('neckBasePoint');
        locs_base((i*K +1):(i+1)*K, :) = imgAugmented.('tailBasePoint');
        difficulty_level((i*K +1):(i+1)*K, 1) = imgAugmented.('difficultyLevelOfImage');
    end
    
    %save data
    uisave({'images', 'direcs_body', 'direcs_head', 'locs_head', 'locs_neck', 'locs_base', 'difficulty_level'}, [filename '_AUG']);
end

function [data, file_name] = load_data()
%load a merge(!) data struct
    [file_name, p_name] = uigetfile('*.mat','Select .mat file');
        if isequal(file_name, 0) || isequal(p_name, 0)
            disp('Action canceled...');
            return;
        end
     data = load(file_name); 
end

function augData = augment_one_image(data, i, K) 
% augment an image, each for loop represents the method that is done
% input:
%   data - a struct of merge data
%   i - index of the specific image in the data
%   K - enlargment factor
    IMG_SIZE = 200;
    SPECKEL_FACTOR = 0.001;
    GAUS_NOISE_VAR_FACTOR = 0.0001;
    POINTS_FACTOR = [0.5, 1]; %range to control the noise of the points
    X = 1;
    Y = 2;
    
    counter = 1;
    im = data.('image')(:,:,i);
    body_angle = data.('bodyAngle')(i);
    head_angle = data.('headAngle')(i);
    head_loc = data.('headPoint')(i,:);
    neck_loc = data.('neckBasePoint')(i,:);
    tail_loc = data.('tailBasePoint')(i,:);
    dLevel = data.('difficultyLevelOfImage')(i);
    
    %augmented data vectors
    DiffLevels = zeros(K, 1) + dLevel; %all difficulty levels of the augmented images are the same
    Images = zeros(IMG_SIZE, IMG_SIZE, K, 'uint8');
    body_angles_vec = zeros(K, 1);
    head_angles_vec = zeros(K, 1);
    head_loc_vec = zeros(K, 2);
    neck_loc_vec = zeros(K, 2);
    tail_loc_vec = zeros(K, 2);
    
   
    for rotateIdx = 1:4%4
        for flipUdIdx = 1:2
            loop2Image = im; %save the image for future use
            for noise2imageIdx = 1:2% can be up to 4
                im = noise2Image(noise2imageIdx, loop2Image, SPECKEL_FACTOR, GAUS_NOISE_VAR_FACTOR);
                loop3Image = im; %save the image for future use
                for jitterIdx = 1:2 % can be up to 8
                    im = jitterImage(jitterIdx, loop3Image, IMG_SIZE);
                    for noise2labelIdx = 1:2
                        [body_angle_org, head_angle_org, head_loc_org, neck_loc_org, tail_loc_org] = deal(body_angle, head_angle, head_loc, neck_loc, tail_loc);%save the orginal tags for future use
                        [body_angle_new, head_angle_new, head_loc_new, neck_loc_new, tail_loc_new] = noise2labels(noise2labelIdx,POINTS_FACTOR ,body_angle_org, head_angle_org, head_loc_org, neck_loc_org, tail_loc_org);

                        %save values of image and label to output matrixs
                        Images(:,:, counter) = im;
                        body_angles_vec(counter) = body_angle_new;
                        head_angles_vec(counter) = head_angle_new;
                        head_loc_vec(counter, :) = [head_loc_new(X), head_loc_new(Y)];
                        neck_loc_vec(counter, :) = [neck_loc_new(X), neck_loc_new(Y)];
                        tail_loc_vec(counter, :) = [tail_loc_new(X), tail_loc_new(Y)];
                        counter = counter + 1;
                        %////////////////////////////////////////////
                    end
                    
                end
            end
            %flipud code - flip image, change angles, change points.
            im = flipud(im);

            [head_loc(X), head_loc(Y)] = rotateUd(head_loc(X), head_loc(Y));
            [neck_loc(X), neck_loc(Y)] = rotateUd(neck_loc(X), neck_loc(Y));
            [tail_loc(X), tail_loc(Y)] = rotateUd(tail_loc(X), tail_loc(Y));
            
            body_angle = (wrapTo360(rad2deg(atan2(tail_loc(Y)-neck_loc(Y), neck_loc(X)-tail_loc(X)))));
            head_angle = (wrapTo360(rad2deg(atan2(neck_loc(Y)-head_loc(Y), head_loc(X)-neck_loc(X)))));
        end
        %rotate code -  rotate image, change angles, change points
        im = rot90(im);

        [head_loc(X), head_loc(Y)]= rotateXY90(head_loc(X), head_loc(Y));
        [neck_loc(X), neck_loc(Y)]= rotateXY90(neck_loc(X), neck_loc(Y));
        [tail_loc(X), tail_loc(Y)]= rotateXY90(tail_loc(X), tail_loc(Y));
        
        body_angle = (wrapTo360(rad2deg(atan2(tail_loc(Y)-neck_loc(Y), neck_loc(X)-tail_loc(X)))));
        head_angle = (wrapTo360(rad2deg(atan2(neck_loc(Y)-head_loc(Y), head_loc(X)-neck_loc(X)))));
    end
    
    augData = struct('image', Images,...
            'headAngle', head_angles_vec,...
            'bodyAngle', body_angles_vec,...
            'headPoint', head_loc_vec,...
            'neckBasePoint', neck_loc_vec,...
            'tailBasePoint', tail_loc_vec,...
            'difficultyLevelOfImage', DiffLevels);
end

% ============= Helper functions =======================

function [x, y] = rotateXY90(x , y)
%rotating x,y values
    IMG_SIZE = 200;
    %rotate value, using a temp
    temp = y;
    y = -x;
    x = temp;
    %check if the point should be fix accoriding to image boundaries
    if x < 0
        x = x + IMG_SIZE;
    end
    if y < 0
        y = y + IMG_SIZE;
    end
end

function [x, y] = rotateUd(x, y)
    %flip up side down for the head and body coordinates.
    IMG_SIZE = 200;
    y = IMG_SIZE - y;
end

function im = noise2Image(idx, originalIm, SPECKLE_FACTOR, GAUS_NOISE_VAR_FACTOR)
%adds noise to image, each iteration different noise is added to the original
%image! detailes of the noises in matlab imnoise
    if idx == 1
        im = imnoise(originalIm,'poisson');
    end
    if idx == 2
        im = imnoise(originalIm,'gaussian',0, GAUS_NOISE_VAR_FACTOR);
    end
    if idx == 3
        im = imnoise(originalIm,'speckle', SPECKLE_FACTOR);
    end
    if idx == 4
        im = originalIm;
    end
end

function[body_angle, head_angle, head_loc, neck_loc, tail_loc] = noise2labels(idx, pointsF ,bAngOrg, hAngOrg, hPOrg, nPOrg, tPOrg)
    %adds noise to the label, adds randomale noise to the locations of the
    %points (head, neck and tail) and calculate the new angles.
    X = 1;
    Y = 2;
    [body_angle, head_angle, head_loc, neck_loc, tail_loc] = deal(bAngOrg, hAngOrg, hPOrg, nPOrg, tPOrg) ; %multiple assignment
    
    if idx == 2
        noise = (pointsF(2)-pointsF(1)).*rand(3,2) + pointsF(1);
        head_loc = noise(1) + head_loc;
        neck_loc = noise(2) + neck_loc;
        tail_loc = noise(3) + tail_loc;
        body_angle = (wrapTo360(rad2deg(atan2(tail_loc(Y)-neck_loc(Y), neck_loc(X)-tail_loc(X)))));
        head_angle = (wrapTo360(rad2deg(atan2(neck_loc(Y)-head_loc(Y), head_loc(X)-neck_loc(X)))));
    end
end

function im = jitterImage(idx, originalIm, IMG_SIZE)
    % jitter- moves the whole image each time by a row\col\both but the tag
    % stays the same! each iteration create different image by a different
    % move. 
    im = originalIm;
    if idx == 1
        im = originalIm;
    end
    if idx == 2
        % a move left
        im(1:IMG_SIZE-1, :) = originalIm(2:IMG_SIZE, :);
    end
    if idx == 3
        % a move up
        im(:,1:IMG_SIZE-1) = originalIm(:,2:IMG_SIZE);
    end
    if idx == 4
        % a move right
        im(2:IMG_SIZE, :) = originalIm(1:IMG_SIZE-1, :);
    end
    if idx == 5
        % a move down
        im(:,2:IMG_SIZE) = originalIm(:,1:IMG_SIZE-1);
    end
    if idx == 6
        % a move left and up
        im(1:IMG_SIZE-1, :) = originalIm(2:IMG_SIZE, :);
        im(:,1:IMG_SIZE-1) = im(:,2:IMG_SIZE);
    end
    if idx == 7
        % a move left and down
        im(1:IMG_SIZE-1, :) = originalIm(2:IMG_SIZE, :);
        im(:,2:IMG_SIZE) = im(:,1:IMG_SIZE-1);
    end
    if idx == 8
        % a move right and up
        im(2:IMG_SIZE, :) = originalIm(1:IMG_SIZE-1, :);
        im(:,1:IMG_SIZE-1) = im(:,2:IMG_SIZE);
    end
end