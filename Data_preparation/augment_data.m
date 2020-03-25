function augment_data()
    %augment a choosen images file, each image turns into K images
    [data,filename] = load_data();
    IMG_SIZE = 200;
    K = 64; %enlargement factor
    numOfImages = size(data.('image'),3);
    augSize = numOfImages*K;
    
    %intilaizing new Data matrixs
    images = zeros(IMG_SIZE, IMG_SIZE, augSize, 'uint8');
    direcs_body = zeros(augSize, 1);
    direcs_head = zeros(augSize, 1);
    locs_head = zeros(augSize, 2);
    locs_neck = zeros(augSize, 2);
    locs_base = zeros(augSize, 2);
    difficulty_level = zeros(augSize, 1);
    %insert basic\old data to matrixs
    
    for i = 0:(numOfImages-1) %check +1!!!!!!!!
        imgAugmented = augment_one_image(data, i+1, K);
        %insert data of augmenting one image to matrixs
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
%augment an image, each for loop represents the change in the image/label
%input - data - a struct of merge data
%i - index of the specific image in the data
% K - enlargment factor
    IMG_SIZE = 200;
    SPECKEL_FACTOR = 0.001;
    GAUS_NOISE_VAR_FACTOR = 0.0001;
    POINTS_FACTOR = [0.5, 1]; %range to control the noise of the points
    X = 1;
    Y = 2;
    
    counter = 1;
    im = data.('image')(:,:,i);
    bAng = data.('bodyAngle')(i);
    hAng = data.('headAngle')(i);
    hP = data.('headPoint')(i,:);
    nP = data.('neckBasePoint')(i,:);
    tP = data.('tailBasePoint')(i,:);
    dLevel = data.('difficultyLevelOfImage')(i);
    
    DiffLevels = zeros(K, 1) + dLevel; %all difficulty levels of the augmented images are the same
    Images = zeros(IMG_SIZE, IMG_SIZE, K, 'uint8');
    BodyAngels = zeros(K, 1);
    HeadAngels = zeros(K, 1);
    HeadPoints = zeros(K, 2);
    NeckPoints = zeros(K, 2);
    TailPoints = zeros(K, 2);
    
   
    for rotateIdx = 1:4%4
        for flipUdIdx = 1:2
            loop2Image = im; %save the image for future use
            for noise2imageIdx = 1:2%4
                im = noise2Image(noise2imageIdx, loop2Image, SPECKEL_FACTOR, GAUS_NOISE_VAR_FACTOR);
                loop3Image = im; %save the image for future use
                for jitterIdx = 1:2 %8
                    im = jitterImage(jitterIdx, loop3Image, IMG_SIZE);
                    for noise2labelIdx = 1:2
                        [bAngOrg, hAngOrg, hPOrg, nPOrg, tPOrg] = deal(bAng, hAng, hP, nP, tP);%save the orginal tags for future use
                        [bAngNew, hAngNew, hPNew, nPNew, tPNew] = noise2labels(noise2labelIdx,POINTS_FACTOR ,bAngOrg, hAngOrg, hPOrg, nPOrg, tPOrg);

                        %save values of image and label to output matrixs
                        Images(:,:, counter) = im;
                        BodyAngels(counter) = bAngNew;
                        HeadAngels(counter) = hAngNew;
                        HeadPoints(counter, :) = [hPNew(X), hPNew(Y)];
                        NeckPoints(counter, :) = [nPNew(X), nPNew(Y)];
                        TailPoints(counter, :) = [tPNew(X), tPNew(Y)];
                        counter = counter + 1;
                        %////////////////////////////////////////////
                    end
                    
                end
            end
            %flipud code - flip image, change angles, change points.
            im = flipud(im);

            [hP(X), hP(Y)] = rotateUd(hP(X), hP(Y));
            [nP(X), nP(Y)] = rotateUd(nP(X), nP(Y));
            [tP(X), tP(Y)] = rotateUd(tP(X), tP(Y));
            
            bAng = (wrapTo360(rad2deg(atan2(tP(Y)-nP(Y), nP(X)-tP(X)))));
            hAng = (wrapTo360(rad2deg(atan2(nP(Y)-hP(Y), hP(X)-nP(X)))));
        end
        %rotate code -  rotate image, change angles, change points
        im = rot90(im);

        [hP(X), hP(Y)]= rotateXY90(hP(X), hP(Y));
        [nP(X), nP(Y)]= rotateXY90(nP(X), nP(Y));
        [tP(X), tP(Y)]= rotateXY90(tP(X), tP(Y));
        
        bAng = (wrapTo360(rad2deg(atan2(tP(Y)-nP(Y), nP(X)-tP(X)))));
        hAng = (wrapTo360(rad2deg(atan2(nP(Y)-hP(Y), hP(X)-nP(X)))));
    end
    
    augData = struct('image', Images,...
            'headAngle', HeadAngels,...
            'bodyAngle', BodyAngels,...
            'headPoint', HeadPoints,...
            'neckBasePoint', NeckPoints,...
            'tailBasePoint', TailPoints,...
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
    %rotating x,y value upside down
    IMG_SIZE = 200;
%     x = IMG_SIZE - x;
    y = IMG_SIZE - y;
end

function im = noise2Image(idx, originalIm, SPECKLE_FACTOR, GAUS_NOISE_VAR_FACTOR)
%adds noise to image, each round differnt noise is added to the original
%image! detailes on the noise in matlab imnoise
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

function[bAng, hAng, hP, nP, tP] = noise2labels(idx, pointsF ,bAngOrg, hAngOrg, hPOrg, nPOrg, tPOrg)
    %adds noise to the label, adds randomale noise to the locations of the
    %points (head, neck and tail) and calculate the new angles.
    X = 1;
    Y = 2;
    [bAng, hAng, hP, nP, tP] = deal(bAngOrg, hAngOrg, hPOrg, nPOrg, tPOrg) ; %multiple assignment
    
    if idx == 2
        noise = (pointsF(2)-pointsF(1)).*rand(3,2) + pointsF(1);
        hP = noise(1) + hP;
        nP = noise(2) + nP;
        tP = noise(3) + tP;
        bAng = (wrapTo360(rad2deg(atan2(tP(Y)-nP(Y), nP(X)-tP(X)))));
        hAng = (wrapTo360(rad2deg(atan2(nP(Y)-hP(Y), hP(X)-nP(X)))));
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