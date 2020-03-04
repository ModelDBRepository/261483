
function Sim(experimentFlag)

    %###########################################################
    %##################### set  parameters ###########################
    %###########################################################

    rng(1337); % set random seed
    
    resultsFolder = './results';
    savePath = sprintf('%s/%s_%s/', resultsFolder, datestr(now, 'yy-mm-dd_HH-MM'), experimentFlag); % might remain the only variable defined here

    % load/Initialize Model
    loadModel = 0;
    loadModelPath = './Path/to/model/';
    loadWeights = 0;    % whether to use previous values for sparse coder or RL agents
    loadBases = 0;      % load bases from separate file independent of loaded model
    preSample = 1;      % are patches presampled before simulation starts? (decreases runtime for large 'T_train')
    loadSample = 0;     % load presampled patches in advance
    saveSample = 0;     % save presampled patches to file which can be reused
    
    % search in resultsFolder for folder with flags, to load samples from
    if ~isempty(strfind(experimentFlag, 'hc'))        
        sample_Flag = 'hc';   
    elseif ~isempty(strfind(experimentFlag, 'aniso'))        
        sample_Flag = 'aniso';
    elseif ~isempty(strfind(experimentFlag, 'mono'))        
        sample_Flag = 'mono';
    else
       sample_Flag = 'error';
    end
    
    %% create new model & copy source files as backup to results folder
    model = config(savePath); %instantiate model object
    mkdir(savePath);
    copyfile(strcat(mfilename, '.m'), savePath);
    copyfile('config.m', savePath);
    copyfile('Model.m', savePath);
    copyfile('SparseCoding.m', savePath);
    copyfile('ReinforcementLearning.m', savePath);
    copyfile('PatchGenerator.m', savePath);
    copyfile('calculateRightBinocularity.m', savePath);
    copyfile('calculateCorrelation.m', savePath);
    copyfile('laprnd.m', savePath);
    copyfile('Gabor.m', savePath)

    if loadModel  % start with a pretrained model
        modelTrained = load(strcat(loadModelPath, '/model.mat'));
        modelTrained = modelTrained.model;

        % load RLs
        if loadWeights
            model.rlmodel_blur = modelTrained.rlmodel_blur;
            model.rlmodel_disp = modelTrained.rlmodel_disp;
        end

        % load SC Basis
        for s = 1:length(model.scmodel)
            model.scmodel{s} = modelTrained.scmodel{s};
            model.scmodel{s}.Basis_hist(:,:,1) = modelTrained.scmodel{s}.Basis;
        end

        % load parameters for reward normalization
        model.blur_reward_mean = modelTrained.blur_reward_mean;
        model.disp_reward_mean = modelTrained.disp_reward_mean;
        model.blur_reward_variance = modelTrained.blur_reward_variance;
        model.disp_reward_variance = modelTrained.disp_reward_variance;

        clearvars modelTrained  %remove loaded model from workspace to free up RAM
    end

    if loadBases        
        data = load('basis_coarse.mat','basis');
        model.scmodel{1}.Basis = data.basis;
        model.scmodel{1}.Basis_hist(:, :, 1) = data.basis;
        data = load('basis_fine.mat','basis');
        model.scmodel{2}.Basis = data.basis;
        model.scmodel{2}.Basis_hist(:, :, 1) = data.basis;

        copyfile('basis_coarse.mat', savePath);
        copyfile('basis_fine.mat', savePath);
    end

    % prepare parameters for patch generation
    nScales = length(model.scmodel);
    params = {};
    for s = 1:nScales
        params{s} = {model.scmodel{s}.patch_size, model.scmodel{s}.Dsratio, model.blur_scale, model.cataract_l, model.cataract_r};
    end

    %###############################################################
    %################# initialize or load input patches ########################
    %###############################################################

    files = dir([model.texture_directory '*.bmp']);

    if (loadSample && ~exist('sample_Left','var') && ~exist('sample_Right','var'))
        loadingPath = dir(sprintf('%s/*%s-samples',resultsFolder, sample_Flag));
        loaded = 0;
        for directory = 1 : length(loadingPath)
            if loaded
                continue
            end
            try
                display('Load patches_Left.mat');
                load(strcat(resultsFolder, '/' , loadingPath(directory).name, '/patches_Left.mat'));
                load(strcat(resultsFolder, '/' , loadingPath(directory).name, '/rewardsL.mat'));
                display('Load patches_Right.mat');
                load(strcat(resultsFolder, '/' , loadingPath(directory).name, '/patches_Right.mat'));
                load(strcat(resultsFolder, '/' , loadingPath(directory).name, '/rewardsR.mat'));
                display('patches loaded!');
                loaded = 1;
            catch
            end
        end
        if ~loaded
            sprintf('loading failed')
        end
    elseif (~loadSample)
        sample_Left = {};
        sample_Right = {};
        accRewLeft = {};
        accRewRight = {};
        texture_images = {};
        tic;

        for k = 1 : model.n_textures
            img = imread([model.texture_directory files(k).name]);
            if(model.colored_input_images==1)
                img = .2989*img(:,:,1) +.5870*img(:,:,2) +.1140*img(:,:,3); %RGB to grayscale
            end
            img=im2double(img);
            if(size(img,1)~=model.image_size || size(img,2)~=model.image_size)
                img=imcrop(img,[(size(img,1)/2-model.image_size/2) (size(img,2)/2-model.image_size/2) model.image_size-1 model.image_size-1]);
            end
            if(size(img,1)~=model.image_size || size(img,2)~=model.image_size)
                disp('error - incorrect format of input image');
            end
            texture_images{k}=img;
            if preSample
                for blur=-model.maxBlur:1:model.maxBlur
                    for dispa=-model.maxDisp:1:model.maxDisp
                        for s = 1:nScales
                            [patchesLeft, patchesRight, accRewL, accRewR] = PatchGenerator(texture_images{k}, dispa, blur-model.spectacles_l, blur-model.spectacles_r, model, params{s});
                            sample_Left{k, s}{dispa+model.maxDisp+1, blur+model.maxBlur+1} = patchesLeft;
                            sample_Right{k, s}{dispa+model.maxDisp+1, blur+model.maxBlur+1} = patchesRight;
                            accRewLeft{k, s}{dispa+model.maxDisp+1, blur+model.maxBlur+1} = accRewL;
                            accRewRight{k, s}{dispa+model.maxDisp+1, blur+model.maxBlur+1} = accRewR;
                        end
                    end
                end
            end
            display(['texture: ' num2str(k)]); %print rendering progress
        end

        save(strcat(savePath, 'rewardsL.mat'), 'accRewLeft');
        save(strcat(savePath, 'rewardsR.mat'), 'accRewRight');
        toc;
    end

    if (preSample && saveSample)
        display('Save patches_Left.mat');
        save(strcat(savePath, 'patches_Left.mat'),'sample_Left','-v7.3');
        display('Save patches_Right.mat');
        save(strcat(savePath, 'patches_Right.mat'),'sample_Right','-v7.3');
        display('patches saved!');
    end

    
    %###############################################################
    %################# initializing simulation variables ########################
    %###############################################################

    current_texture = randi(model.n_textures);

    %generate object positions uniformly between -3 to 3
    model.distances = randi([-model.maxObjPlane model.maxObjPlane], 1, model.T_train/model.Interval + 1);
    index = 1;
    curr_disp = model.distances(index);
    curr_blur = model.distances(index);
    blurIndexOffset = ceil(model.rlmodel_blur.Action_num / 2);
    dispIndexOffset = ceil(model.rlmodel_disp.Action_num / 2);

    shift = 0;              % initial vergence plane
    lense_blur = 0;         % initial accommodation plane
    monocularity = 0.5;     % initial binocularity index: ranges from -1 (left mon.) to 1 (right mon.)
    contrastLeft = 1;
    contrastRight = 1;
    nFeatVec = 0;           % for initializing the feature vector
    for s = 1:nScales
        nFeatVec = nFeatVec + model.scmodel{s}.Basis_num;
    end
    feature = abs(rand(nFeatVec, 1) - 0.5);     % zero mean init of coeffitients to enable suppression from beginning
    accRewL = {};
    accRewR = {};
    patchesLeft = {};
    patchesRight = {};
    monDetectLeft = 0;
    monDetectRight = 0;
    monDetectLeft = 0;
    monDetectRight = 0;

    runningAvgLC = 0.5;  % initialization for the moving averages
    runningAvgRC = 0.5;
    runningAvgLF = 0.5;
    runningAvgRF = 0.5;

    %% -
    %###########################################################################
    %######################## MAIN SIMULATION ##################################
    %###########################################################################

    tic
    done = 0;                % controle parameter if sim is finished
    t = model.trainedUntil;  % current # of iterations, starts with 0
    while(~done)
        
        % display training process and save model 11 times per simulation
        if(find(t == model.saveAt))
            model.trainedUntil = t;

            disp([num2str(t/model.T_train*100) '% is finished']);
            if model.trainSC
                % save basis
                for s = 1:nScales
                    model.scmodel{s}.saveBasis(find(t==model.saveAt));
                end
            end

            if model.trainRL
                % save weights
                model.rlmodel_blur.saveWeights(find(t==model.saveAt));  %save policy and value net weights
                model.rlmodel_disp.saveWeights(find(t==model.saveAt));  %save policy and value net weights
            end

            % save model
            save(strcat(savePath, 'model.mat'),'model');
        end

        % update variables
        t = t+1;    % iteration counter
        if (t >= model.T_train)
            done = 1;
        end
        
        % change object and its position every 'Interval' iterations
        if ~mod(t,model.Interval)
            index = index + 1;
            current_texture = randi(model.n_textures,1);  % random texture
            curr_disp = model.distances(index);
            curr_blur = model.distances(index);
        end

        if (preSample || loadSample)
            blur = curr_blur - lense_blur + model.maxBlur + 1;   % transform to indices
            dispa = curr_disp - shift + model.maxDisp + 1;

            for s = 1:nScales
                patchesLeft{s} = sample_Left{current_texture, s}{dispa, blur};
                patchesRight{s} = sample_Right{current_texture, s}{dispa, blur};
                accRewL{s} = accRewLeft{current_texture, s}{dispa, blur};
                accRewR{s} = accRewRight{current_texture, s}{dispa, blur};
            end
        else  % online sampling
            d = curr_disp - shift;          % disparity
            b = curr_blur - lense_blur;     % blur
            for s = 1:nScales
                [patchesLeft{s}, patchesRight{s}, accRewL{s}, accRewR{s}] = PatchGenerator(texture_images{current_texture}, d, b-model.spectacles_l, b-model.spectacles_r, model, params{s});
            end
        end

        %#######################################################################
        %#################### Suppression Mechanism ############################
        %#######################################################################

        if model.useSuppression
            for s = 1:nScales
                binocularity = calculateRightBinocularity(model.scmodel{s}.Basis);      % weights to contrast units
                                                                                        % 0: left monocular, 0.5: binocular, 1: right monocular
                if s == 1                                                               
                    activities = feature(1:model.scmodel{s}.Basis_num);                 % feature vector is already normalized since feature vector is normalized
                elseif s == 2
                    activities = feature(model.scmodel{s}.Basis_num+1 : end);
                end

                monDetectLeft = dot(activities, (1 - binocularity));                    % weight binocularities by cortical cell activity
                monDetectRight = dot(activities, binocularity);

                if s == 1
                    runningAvgLC = runningAvgLC + model.eta_supp * (monDetectLeft - runningAvgLC);  % running average of contrast measure
                    runningAvgRC = runningAvgRC + model.eta_supp * (monDetectRight - runningAvgRC);

                    model.monocularity_hist(t, 3) = runningAvgLC;
                    model.monocularity_hist(t, 4) = runningAvgRC;

                    monDetectLeft = max(0, model.slope * (runningAvgLC - model.threshold));  % output of contrast units dependent on slope and threshold
                    monDetectRight = max(0, model.slope * (runningAvgRC- model.threshold));

                    contrastLeft = 1 + model.exci*monDetectLeft - model.suppr*monDetectRight;
                    contrastRight = 1 - model.suppr*monDetectLeft + model.exci*monDetectRight;

                    model.contrast_hist(t, 1) = contrastLeft;  % record contrast values
                    model.contrast_hist(t, 2) = contrastRight;
                    model.monocularity_hist(t, 5) = monDetectLeft;
                    model.monocularity_hist(t, 6) = monDetectRight;

                elseif s == 2
                    runningAvgLF = runningAvgLF + model.eta_supp * (monDetectLeft - runningAvgLF);
                    runningAvgRF = runningAvgRF + model.eta_supp * (monDetectRight - runningAvgRF);

                    model.monocularity_hist(t, 7) = runningAvgLF;
                    model.monocularity_hist(t, 8) = runningAvgRF;

                    monDetectLeft = max(0, model.slope * (runningAvgLF - model.threshold));  % send through linear activation unit
                    monDetectRight = max(0, model.slope * (runningAvgRF - model.threshold));

                    contrastLeft = 1 + model.exci*monDetectLeft - model.suppr*monDetectRight;
                    contrastRight = 1 - model.suppr*monDetectLeft + model.exci*monDetectRight;

                    model.contrast_hist(t, 3) = contrastLeft;
                    model.contrast_hist(t, 4) = contrastRight;
                    model.monocularity_hist(t, 9) = monDetectLeft;
                    model.monocularity_hist(t, 10) = monDetectRight;
                end

                % adjust contrast of input patches patch adaptation
                patchesLeft{s} = patchesLeft{s} .* contrastLeft;
                patchesRight{s} = patchesRight{s} .* contrastRight;
                
                % adjust acc reward according to left/right eye suppression
                accRewL{s} = accRewL{s} * contrastLeft;  
                accRewR{s} = accRewR{s} * contrastRight; 
            end
        end

        if model.trainRL || loadWeights
            [feature, reward, err, coef, monocularity] = model.generateFR(patchesLeft, patchesRight, t, sum([accRewL{:}]), sum([accRewR{:}]));
        end

        %% TRAINING
        
        % if SC AND RL are initialized: update receptive fields
        if model.trainSC && (model.trainRL || loadWeights )
            for s = 1:nScales
                model.scmodel{s}.updateBasis(coef{s},err{s});
            end
            
        %if no RL, encode input and update receptive fields without suppression  
        elseif model.trainSC
            coefs = [];
            for s = 1:nScales
                [err, coef, monocularity] = model.scmodel{s}.stepTrain([patchesLeft; patchesRight]);
                coefs = vertcat(coefs, coef);
            end
            feature = mean(coefs.^2, 2);
            feature = feature./sum(feature); % normalize feature vector
        end

        % get RL actions and then update RL agents
        if model.trainRL
            action_blur = model.rlmodel_blur.stepTrain(feature,reward(1),mod(t-1,model.Interval),0); %index of action
            action_disp = model.rlmodel_disp.stepTrain(feature,reward(2),mod(t-1,model.Interval),0);
        end

        % if no RL model is available, do not move planes and set reward to zero
        if (~model.trainRL && ~loadWeights)
            action_blur = 0 + blurIndexOffset;
            action_disp = 0 + dispIndexOffset;
            reward=[0 0 0];
        end

        %if RL training is off but RL model loaded, choose greedy action, i.e. no exploration
        if (~model.trainRL && loadWeights)
            action_blur = model.rlmodel_blur.Act(feature);  %returns index of action
            action_disp = model.rlmodel_disp.Act(feature);
            reward=[0 0 0];
        end

        % track some variables
        blur_command = model.Act_blur(action_blur);
        shift_command = model.Act_disp(action_disp);
        model.rlmodel_blur.Action_hist(t) = blur_command;
        model.rlmodel_disp.Action_hist(t) = shift_command;
        model.rlmodel_blur.pol_hist(t, :) = model.rlmodel_blur.pol;
        model.rlmodel_disp.pol_hist(t, :) = model.rlmodel_disp.pol;
        model.rlmodel_blur.td_hist(t) = model.rlmodel_blur.td;
        model.rlmodel_disp.td_hist(t) = model.rlmodel_disp.td;
        model.recerr_hist(t, :) = mean(sum([err{:}].^2));  % reconstruction error
        d = curr_disp - shift;          % input disparity
        b = curr_blur - lense_blur;     % defocus blur value
        model.vergerr_hist(t) = d;
        model.accerr_hist(t) = b;
        model.reward_hist(t,:) = [d; b; reward(1); reward(2); reward(3); blur_command; shift_command];
        model.texture_hist(t,1) = current_texture;
        model.monocularity_hist(t, 1:nScales) = monocularity;
        model.objPlane_hist(t) = curr_disp;
        model.vergPlane_hist(t) = shift;
        model.accPlane_hist(t) = lense_blur;

        % apply plane position shifts
        shift = shift + shift_command;
        lense_blur = lense_blur + blur_command;

        if ((abs(shift) > model.maxObjPlane))  % check if eye fixation is out of bound
            shift = model.maxObjPlane * sign(shift);  % set fixation at max/min distance
        end
        if (abs(lense_blur) > model.maxAccDispPlane)  % check if focus reaches near/far-point
            lense_blur = model.maxAccDispPlane * sign(lense_blur);  % set focus at max/min distance
        end
    end
    
    TotT = toc/60;  % total simulation time
    sprintf(' Time [min] = %.2f\non avergage %.1f iters/sec',TotT, model.T_train / (TotT*60))
    model.simulationTime = TotT/60;

    model.trainedUntil = t;  % track when model was last saved

    if model.trainSC
        %save basis
        for s = 1:nScales
            model.scmodel{s}.saveBasis(find(t==model.saveAt));
        end
    end
    if model.trainRL
        %save RL weights
        model.rlmodel_blur.saveWeights(find(t==model.saveAt));  %save policy and value net weights
        model.rlmodel_disp.saveWeights(find(t==model.saveAt));  %save policy and value net weights
    end
    
    %save data
    save(strcat(savePath, 'model.mat'),'model');
    
    % plot results
    model.allPlotSave();
end
