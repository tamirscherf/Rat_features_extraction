function net = train_net(images, direcs, repeat_n)
    % This function trains one net or more, and allows changing desired parameters 
    % of the net. It contains two architectures: ResNet and Custom.
    % It create all the validation for each net, graphs and video, and saves
    % the net and those validations of it in a unique folder. 
    
    % ======== Manipulation over repetitions
    % Change the learning rate, learning rate drop period and net type
    % for each iteration of the training section. 
    if repeat_n == 2
        learning_rate = 0.2;
        drop_period =  50;
        net_type = 'resnet';
    else
        learning_rate = 0.2;
        drop_period =  50;
        net_type = 'custom';
        % In custom wide_layer and narrow_layer have been switched!
    end
    % ======== Set net_type and train size.
    NET_TYPE = net_type;  % {'custom', 'resnet'}, types of architectures to use
    TRAIN_SIZE = -1;      % Number of images to use, '-1' for all or '10000' for overfit test
    INPUT_SIZE = size(images, 1);
    
    % ======== Set n_images and image_size
    if TRAIN_SIZE < 0
        n_images = length(direcs);
    else
        n_images = TRAIN_SIZE;
    end
    image_size = size(images, 1);
    
    % ======== Set the order of the images randomly
    rand_sub_set_inds = randsample(length(direcs), n_images);
    direcs = direcs(rand_sub_set_inds);
    images = images(:, :, rand_sub_set_inds);

    % ======== split data into traing and validation
    [train_inds, ~, test_inds] = dividerand(n_images, 0.8, 0.0, 0.2);
    images_test = reshape(images(:, :, test_inds), [image_size, image_size, 1, length(test_inds)]);
    direcs_test = direcs(test_inds);
    images_train = reshape(images(:, :, train_inds), [image_size, image_size, 1, length(train_inds)]);
    direcs_train = direcs(train_inds);

    % ===== networks code ==================
    % ===== RESNET code ==================
    if strcmp(NET_TYPE, 'resnet')        
        net_width = 24;
        lgraph = create_layers_resnet(image_size, net_width);
        batch_size = 64;
        epochs = 200;
        verbose_and_eval_every = 10;  % in epochs
        options = trainingOptions('adam', ...
            'InitialLearnRate',     0.001, ...
            'MaxEpochs',            epochs, ...
            'MiniBatchSize',        batch_size, ...
            'VerboseFrequency',     round(n_images/batch_size)*verbose_and_eval_every, ...
            'Shuffle',              'every-epoch', ...
            'Plots',                'training-progress', ...
            'Verbose',              true, ...
            'LearnRateSchedule',    'piecewise', ...
            'LearnRateDropFactor',  learning_rate, ...
            'LearnRateDropPeriod',  drop_period, ...
            'ValidationData',       {images_test, direcs_test}, ...
            'ValidationFrequency',  round(n_images/batch_size)*verbose_and_eval_every);
        
        net = trainNetwork(images_train, direcs_train, lgraph, options);
    % ===== CUSTOME code ==================
    else
        %NEW LAYERS - down in one
        layers = create_layers_custom_ANN(image_size);
        batch_size = 64;
        epochs = 200;
        verbose_and_eval_every = 10;  % in epochs
        options = trainingOptions('adam', ...
            'InitialLearnRate',     0.001, ...
            'MaxEpochs',            epochs, ...
            'MiniBatchSize',        batch_size, ...
            'VerboseFrequency',     round(n_images/batch_size)*verbose_and_eval_every, ...
            'Shuffle',              'every-epoch', ...
            'Plots',                'training-progress', ...
            'Verbose',              true, ...
            'LearnRateSchedule',    'piecewise', ...
            'LearnRateDropFactor',  learning_rate, ...
            'LearnRateDropPeriod',  drop_period, ...
            'ValidationData',       {images_test, direcs_test}, ...
            'ValidationFrequency',  round(n_images/batch_size)*verbose_and_eval_every);
        net = trainNetwork(images_train, direcs_train, layers, options);
    end

    % =====================
    % ===== Varification code ==================
    date_time = strrep(datestr(datetime), ':', '_');
    shared_dir_name = 'dir_name';
    dir_name = fullfile(shared_dir_name, 'Main_code\Head_Nets_runs_stats', [date_time '_' net_type '_i' num2str(repeat_n)]);
    mkdir(dir_name);
    code_f_path = [mfilename('fullpath') '.m'];
    status = copyfile(code_f_path, dir_name, 'f');
    if status == 0
        disp('>>> Failed to copy the code file! <<<');
        disp(['>>> Source: ' code_f_path ' <<<']);
        disp(['>>> Target: ' dir_name ' <<<']);
    end
    fig = findall(groot, 'Type', 'Figure');
    saveas(fig(1), fullfile(dir_name, 'Training_progress.jpg'), 'jpeg');
    save(fullfile(dir_name, [date_time 'trained_net.mat']), 'net');
    save(fullfile(dir_name, 'data_partition.mat'), 'rand_sub_set_inds', 'train_inds' ,'test_inds');
    %#####show histogram of validation#####
    show_histogram(net.predict(images_test),direcs(test_inds), 'Validation');
    saveas(gcf, fullfile(dir_name, 'Validation_histogram.jpg'),'jpeg');
    
    %#####show histogram of training#####
    show_histogram(net.predict(images_train), direcs(train_inds), 'Training');
    saveas(gcf, fullfile(dir_name, 'Training_histogram.jpg'), 'jpeg');

    %#####show histogram of final test#####
    val_db = load('FINAL_TEST_DATA_FIXED_for head.mat');
   
    % Reshape to 100X100 images
    bin = 2 + 2*(INPUT_SIZE == 50);  % 4 for 50, 2 for 100
    image_val = val_db.images(1:bin:200,1:bin:200,:);
    show_histogram(net.predict(reshape(image_val(:, :, :), [image_size, image_size, 1, 100])),...
                val_db.direcs_head, 'Final test');
    saveas(gcf, fullfile(dir_name, 'Final_test_FIXED_histogram.jpg'), 'jpeg');

    %#### Create un-tagged video frames predictions ####
    db_val_ims = load('file_name');
    bin = 1 + (INPUT_SIZE == 50);  % 2 for 50, 1 for 100
    val_direcs_310119 = predict_direcs(db_val_ims.im_100(1:bin:end, 1:bin:end, :), {net}, 1);
    save(fullfile(dir_name, 'preds_aug_unsmoothed.mat'), 'val_direcs_310119');
    %#### Create video from predictions ####
    create_pred_video(db_val_ims.im_200, smooth_angles(val_direcs_310119, 5), 600, 31700, dir_name);
    create_pred_video(db_val_ims.im_200, smooth_angles(val_direcs_310119, 5), 600, 3800, dir_name);

    % ===== Repetition code ==================
    if (nargin > 2) && (repeat_n > 1)
            net = train_net(images, direcs, repeat_n-1);
    end
end

function lgraph = create_layers_resnet(image_size, net_width)
    % Function create_layers_resnet creates layergraph for a custom written ResNet architecture.
    % 
    % Inputs:
    %     image_size - (scalar) - Input image width, normally 100 or 200
    %     net_width - (scalar) - number of convolution filters at each layer
    % 
    % Outpus:
    %     lgraph - (LayerGraph object) - A layer graph that represents the ResNet ANN
    
    my_reg_layer = F_One_regression_layer('F1'); %set my regression layer
   
    layers = [
        imageInputLayer([image_size image_size 1], 'Name', 'input')
        batchNormalizationLayer('Name', 'BNInp_in')
        convolution2dLayer(3, net_width, 'Padding', 'same', 'Name', 'convInp')
        batchNormalizationLayer('Name', 'BNInp')
        reluLayer('Name', 'reluInp')

        convolutionalUnit(net_width, 1, 'S1U1')
        additionLayer(2, 'Name', 'add11')
        reluLayer('Name', 'relu11')
        convolutionalUnit(net_width, 1, 'S1U2')
        additionLayer(2, 'Name', 'add12')
        reluLayer('Name', 'relu12')

        convolutionalUnit(2*net_width, 2, 'S2U1')
        additionLayer(2, 'Name', 'add21')
        reluLayer('Name', 'relu21')
        convolutionalUnit(2*net_width, 1, 'S2U2')
        additionLayer(2, 'Name', 'add22')
        reluLayer('Name', 'relu22')

        convolutionalUnit(4*net_width, 2, 'S3U1')
        additionLayer(2, 'Name', 'add31')
        reluLayer('Name', 'relu31')
        convolutionalUnit(4*net_width, 1, 'S3U2')
        additionLayer(2, 'Name', 'add32')
        reluLayer('Name', 'relu32')
        
        % Skip this block if training on 50x50
        convolutionalUnit(8*net_width, 2, 'S4U1')
        additionLayer(2, 'Name', 'add41')
        reluLayer('Name', 'relu41')
        convolutionalUnit(8*net_width, 1, 'S4U2')
        additionLayer(2, 'Name', 'add42')
        reluLayer('Name', 'relu42')
        
        averagePooling2dLayer(8, 'Name', 'globalPool')
        fullyConnectedLayer(1, 'Name', 'fc')
        my_reg_layer()
    ];

    lgraph = layerGraph(layers);
    
    lgraph = connectLayers(lgraph, 'reluInp', 'add11/in2');
    
    lgraph = connectLayers(lgraph, 'relu11', 'add12/in2');
    skip1 = [
        convolution2dLayer(1, 2*net_width, 'Stride', 2, 'Name', 'skipConv1')
        batchNormalizationLayer('Name', 'skipBN1')];
    lgraph = addLayers(lgraph, skip1);
    lgraph = connectLayers(lgraph, 'relu12', 'skipConv1');
    lgraph = connectLayers(lgraph, 'skipBN1', 'add21/in2');
    
    lgraph = connectLayers(lgraph, 'relu21', 'add22/in2');
    skip2 = [
        convolution2dLayer(1, 4*net_width, 'Stride', 2, 'Name', 'skipConv2')
        batchNormalizationLayer('Name', 'skipBN2')];
    lgraph = addLayers(lgraph, skip2);
    lgraph = connectLayers(lgraph, 'relu22', 'skipConv2');
    lgraph = connectLayers(lgraph, 'skipBN2', 'add31/in2');

    lgraph = connectLayers(lgraph, 'relu31', 'add32/in2');
    
    % Skip from here on if train on 50x50
    skip3 = [  
        convolution2dLayer(1, 8*net_width, 'Stride', 2, 'Name', 'skipConv3')
        batchNormalizationLayer('Name', 'skipBN3')];
    lgraph = addLayers(lgraph, skip3);
    lgraph = connectLayers(lgraph, 'relu32', 'skipConv3');
    lgraph = connectLayers(lgraph, 'skipBN3', 'add41/in2');
    lgraph = connectLayers(lgraph, 'relu41', 'add42/in2');
end

function layers = create_layers_custom_ANN(image_size)
    my_reg_layer = F_One_regression_layer('F1'); %set my regression layer
    first_layers = 32;
    deeper_layers = 64;
    deepest_layers = 128;
    layers = [
            imageInputLayer([image_size image_size 1])
            batchNormalizationLayer
            convolution2dLayer(3, first_layers, 'Padding', 'same')
            batchNormalizationLayer
            reluLayer

            maxPooling2dLayer(2,'Stride',2)
            convolution2dLayer(3, first_layers, 'Padding', 'same')
            batchNormalizationLayer
            reluLayer

            averagePooling2dLayer(2,'Stride',1)
            convolution2dLayer(3, deeper_layers, 'Padding', 'same')
            batchNormalizationLayer
            reluLayer

            averagePooling2dLayer(2,'Stride',1)
            convolution2dLayer(3, deeper_layers, 'Padding', 'same')
            batchNormalizationLayer
            reluLayer

            maxPooling2dLayer(2,'Stride',2)
            convolution2dLayer(3, deeper_layers, 'Padding', 'same')
            batchNormalizationLayer
            reluLayer
            
            averagePooling2dLayer(2,'Stride',1)
            convolution2dLayer(3, deeper_layers, 'Padding', 'same')
            batchNormalizationLayer
            reluLayer

            averagePooling2dLayer(2,'Stride',1)
            convolution2dLayer(3, deeper_layers, 'Padding', 'same')
            batchNormalizationLayer
            reluLayer
            
            maxPooling2dLayer(2,'Stride',2)
            convolution2dLayer(3, deeper_layers, 'Padding', 'same')
            batchNormalizationLayer
            reluLayer
            
            averagePooling2dLayer(2,'Stride',1)
            convolution2dLayer(3, deepest_layers, 'Padding', 'same')
            batchNormalizationLayer
            reluLayer

            averagePooling2dLayer(2,'Stride',1)
            convolution2dLayer(3, deepest_layers, 'Padding', 'same')
            batchNormalizationLayer
            reluLayer
            
            maxPooling2dLayer(2,'Stride',2)
            convolution2dLayer(3, deepest_layers, 'Padding', 'same')
            batchNormalizationLayer
            reluLayer

            fullyConnectedLayer(1)
            my_reg_layer()
            ];
end

function layers = convolutionalUnit(n_filters, stride, tag)

    % Function convolutionalUnit defines a single ANN unit of two CNN layers.
    % 
    % Inputs:
    %     n_filters - (scalar) - Number of filters per conv layer
    %     stride - (scalar) - stride of the convolution
    %     tag - (string) - Used to form a unique name
    % 
    % Outputs:
    %     layers - (5x1) layers - Layers that form that specific unit
    layers = [
        convolution2dLayer(3,n_filters,'Padding','same','Stride',stride,'Name',[tag,'conv1'])
        batchNormalizationLayer('Name',[tag,'BN1'])
        reluLayer('Name',[tag,'relu1'])
        convolution2dLayer(3,n_filters,'Padding','same','Name',[tag,'conv2'])
        batchNormalizationLayer('Name',[tag,'BN2'])
    ];
end