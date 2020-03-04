classdef Model < handle
    properties
        
        scmodel;        % SparseCoding class
        rlmodel_blur;   % ReinforcementLearning class
        rlmodel_disp;   % ReinforcementLearning class
        Act_blur;       % action range for blur agent
        Act_disp;       % action space for diparity agent

        % simulation parameters
        T_train;            % total number of interations in the training
        Interval;           % period to change a new environment for the eye
        configpath;         % configuration file path
        trainedUntil;       % in case the training was disrupted, how long was the model trained?
        savePath;           % results file path
        maxObjPlane;        % maximal distance of the object plane
        maxAccDispPlane;    % maximal distance of the accomodation plane
        maxDisp;            % maximally possible disparity
        maxBlur;            % maximally possible blur
        saveAt;             % time points where the model data is saved
        nscales;            % number of scales, i.e., number of sparse coders
        useSuppression;     % bool. number indicating the use of suppressive mechanisms
        threshold;          % threshold of contrast units
        slope;              % slope of contrast units
        eta_supp;           % update rate of contrast estimates
        suppr;              % suppressive effect of suppression model
        exci;               % excitatory effect of suppression model
        simulationTime;     % number of hours, the model was trained
        window_size;        % size of initial window cutout before disparities are applied
        trainRL;            % training of reinforcement learning
        trainSC;            % training of sparse coder
        n_textures;         % number of input images used during training
        texture_directory;  % where the input images are stored
        image_size;         % size of input images
        colored_input_images;% gray values or rgb

        %% record history of values
        recerr_hist;        % history of rec error
        reward_hist;        % history of reward
        vergerr_hist;       % history of vergence error
        accerr_hist;        % history of accommodation error
        vergPlane_hist;     % history of vergence plane
        accPlane_hist;      % history of accomodation plane
        objPlane_hist;      % history of object position
        texture_hist;       % history of which texture was shown
        feature_hist;       % history of base activation during learning
        monocularity_hist;  % history of monocularity of chosen basis functions
        contrast_hist;      % history of contrast adjustments to right and left eye
        If_avg_right_hist   % history of average freq spectrum right
        If_avg_left_hist    % history of average freq spectrum left
        distances;          %object distances during simulation

        %% model data
        blur_reward_mean;    % moving average of reward before normalization
        blur_reward_variance;% moving average of variance of reward before normalization
        disp_reward_mean;    % moving average of reward before normalization
        disp_reward_variance;% moving average of variance of reward before normalization
        alpha;               % update rate for reward moving averages
        blur_scale;          % set relation between 1 [a.u.] of accommodation error and stdv of defocus blur filter

        % simulate visual impairment (anisometropy + cataract)
        spectacles_l;
        spectacles_r;
        cataract_l;
        cataract_r;
    end

    methods
        function obj = Model(PARAM)
            
            obj.nscales = PARAM{2}{1}{7};
            
            obj.configpath = PARAM{1}{1};
            obj.trainSC = PARAM{1}{17};
            obj.trainRL = PARAM{1}{9};
            
            obj.T_train = PARAM{1}{2};
            obj.Interval = PARAM{1}{3};
            obj.trainedUntil = 0;
            obj.savePath = PARAM{1}{8};
            obj.maxObjPlane = PARAM{1}{22};
            obj.maxAccDispPlane = PARAM{1}{23};
            obj.maxDisp = PARAM{1}{10};
            obj.maxBlur = PARAM{1}{11};
            obj.saveAt = PARAM{1}{12};
            
            obj.n_textures = PARAM{1}{18};
            obj.texture_directory = PARAM{1}{19};
            obj.image_size = PARAM{1}{20};
            obj.window_size = PARAM{1}{14};
            obj.colored_input_images = PARAM{1}{21};
            obj.blur_scale = PARAM{1}{16};
            
            obj.useSuppression = PARAM{1}{13}{1};
            obj.threshold = PARAM{1}{13}{2};
            obj.slope = PARAM{1}{13}{3};
            obj.eta_supp = PARAM{1}{13}{4};
            obj.suppr = PARAM{1}{13}{5};
            obj.exci = PARAM{1}{13}{6};
            
            obj.blur_reward_mean=0;
            obj.disp_reward_mean=0;
            obj.blur_reward_variance=1;
            obj.disp_reward_variance=1;
            obj.alpha = PARAM{1}{15};
            
            obj.recerr_hist = zeros(obj.T_train, 1);          % coarse and fine scale reconstruction error
            obj.reward_hist = zeros(obj.T_train, 7);          % coarse and fine scale reward
            obj.vergerr_hist = zeros(obj.T_train, 1);         % saved vergence values
            obj.accerr_hist = zeros(obj.T_train, 1);          % saved blur values
            obj.vergPlane_hist = zeros(obj.T_train, 1);       % save history of verg. plane positions
            obj.accPlane_hist = zeros(obj.T_train, 1);        % save history of acc. plane positions    
            obj.objPlane_hist = zeros(obj.T_train, 1);        % save history of obj. plane positions
            obj.texture_hist = zeros(obj.T_train, 1);         % texture history
            obj.monocularity_hist = zeros(obj.T_train, 5);    % history of monocularity of chosen basis
            obj.contrast_hist = ones(obj.T_train, 2);         % left and right contrast adjustments
            obj.distances = zeros(1,PARAM{1}{2}/PARAM{1}{3}); % holds object distances

            obj.spectacles_l = PARAM{1}{4};
            obj.spectacles_r = PARAM{1}{5};
            obj.cataract_l = PARAM{1}{6};
            obj.cataract_r = PARAM{1}{7};

            %init sparse coders
            for s = 1:obj.nscales
                obj.scmodel{s} = SparseCoding(PARAM{2}{s}, size(PARAM{1}{12}, 2));
            end

            %init RLs
            obj.rlmodel_blur = ReinforcementLearning(PARAM{3}, PARAM{1}{2}, size(PARAM{1}{12}, 2));
            obj.rlmodel_disp = ReinforcementLearning(PARAM{4}, PARAM{1}{2}, size(PARAM{1}{12}, 2));
            obj.Act_blur = PARAM{3}{11};
            obj.Act_disp = PARAM{4}{11};
        end
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%
        %%% Generate the feature vector, rewards and encoding errors.
        %%% Also returns current monocularity of receptive fields and coef
        %%% of cortical neurons after encoding.
        %%% 'feature' is the concatenation of foveal and peripheral scale
        %%% 'reward' is the sum of rewards for foveal and peripheral scale
        %%% 'error' is the sum of errors of foveal and peripheral scale
        %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [feature, reward, err, coef_sparse, monocularity] = generateFR(this, patchesL, patchesR, t, blurRewL, blurRewR)
            
            if length(this.scmodel) == 1
                
                [coef_sparse, err, monocularity] = this.scmodel{1}.sparseEncode([patchesL{1}; patchesR{1}]);
                feature = mean(coef_sparse.^2, 2);
                feature = feature./sum(feature);
                coef_sparse = {coef_sparse};
                rew = {mean(sum(err.^2))};
                
            elseif length(this.scmodel) == 2
                
                % peripheral scale
                [coef_coarse, error_coarse, monocularityC] = this.scmodel{1}.sparseEncode([patchesL{1}; patchesR{1}]);
                featureCoarse = mean(coef_coarse.^2, 2);
                featureCoarse = featureCoarse./sum(featureCoarse);
                rewCoarse = -mean(sum(error_coarse.^2));
                
                % foveal scale
                [coef_fine, error_fine, monocularityF] = this.scmodel{2}.sparseEncode([patchesL{2}; patchesR{2}]);
                featureFine = mean(coef_fine.^2, 2);
                featureFine = featureFine./sum(featureFine);
                rewFine = -mean(sum(error_fine.^2));

                err = {error_coarse; error_fine};
                rew = rewCoarse + rewFine;
                feature = [featureCoarse; featureFine];
                monocularity = [monocularityC; monocularityF];
                coef_sparse = {coef_coarse; coef_fine};
            end
    
            % combine rewards and divide by 100 in order to get small
            % reward at beginning of simulation that grows to unit variance
            % due to online normalization. This prevents instabilities due
            % to random initialization.
            disp_reward = rew/100;                 
            blur_reward = (blurRewL + blurRewR)/100;

            % zero mean unit variance reward via exponentially weighted running average
            r = (blur_reward-this.blur_reward_mean)/(sqrt(this.blur_reward_variance) + eps);
            
            delta=blur_reward-this.blur_reward_mean;
            this.blur_reward_mean = this.blur_reward_mean+this.alpha*delta;
            this.blur_reward_variance = (1-this.alpha)*this.blur_reward_variance+this.alpha*delta^2;

            blur_reward = r;

            r = (disp_reward-this.disp_reward_mean)/(sqrt(this.disp_reward_variance) + eps);
            delta=disp_reward-this.disp_reward_mean;
            this.disp_reward_mean = this.disp_reward_mean+this.alpha*delta;
            this.disp_reward_variance = (1-this.alpha)*this.disp_reward_variance+this.alpha*delta^2;
            
            disp_reward = r;

            reward=[blur_reward disp_reward rew];

        end

        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%
        %%% Plot results 
        %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %% Show receptive fields
        function h = displayBasis(this, scle, M, N, saveTime, savePlot)

            s=sqrt(this.scmodel{scle}.Basis_size/2);
            n = this.scmodel{scle}.Basis_num;

            if isempty(saveTime)
                block = mat2gray(col2im(this.scmodel{scle}.Basis,[s 2*s],[n*s 2*s],'distinct'));
            else
                block = mat2gray(col2im(this.scmodel{scle}.Basis_hist(:, :, saveTime),[s 2*s],[n*s 2*s],'distinct'));
            end
            k=1;
            pic=[];

            for j=1:N
                row=[];
                for i=1:M
                    base_l=block(1+(k-1)*s:k*s,1:s);
                    base_r=block(1+(k-1)*s:k*s,s+1:2*s);
                    brick=[-ones(1,s); base_l; -ones(1,s); base_r; -ones(1,s)]; %left part on top
                    brick=[-ones(2*s+3,1) brick -ones(2*s+3,1)];
                    row=[row brick];
                    k=k+1;
                end

                pic=[pic;row];
            end
            
            figure;
            h = imshow(pic);
            if savePlot
                if scle == 1
                    tag = '_coarse';
                elseif scle == 2
                    tag = '_fine';
                end
                saveas(gca, strcat(this.savePath, 'basis', tag), 'png');
            end

        end

        %% Display the N x M Basis that where choosen for encoding in the last step
        function h = displaySelectedBasis(this, scle, N, M, saveName)

            s = sqrt(this.scmodel{scle}.Basis_size/2);    % single patch size
            n = this.scmodel{scle}.Basis_num;             % total number of basis
            sBox = 2;                                     % size of black box surrounding the basis

            usedB = zeros(this.scmodel{scle}.Basis_num, 1);
            usedB(this.scmodel{scle}.Basis_selected > 0) = 1; % extract indices of basis that where selected by matching pursuit
            inds = find(usedB);
            if isempty(inds)
                sprintf('model contains no selected basis')
                return;
            end

            b = 1;
            h = figure;
            hold on;
            for i = 1:M
                for j = 1:N
                    subplot(M, N, b)

                    img = zeros(2*s + 2*sBox, s + 2*sBox)';    % create the black borders
                    img(sBox+1 : end-sBox, sBox+1 : end-sBox) = mat2gray(reshape(this.scmodel{scle}.Basis(:,inds(b)), [sqrt(size(this.scmodel{1}.Basis,1)/2), sqrt(size(this.scmodel{1}.Basis,1)/2)*2]));
                    imshow(img)
                    title(sprintf('BI = %0.2f', calculateRightBinocularity(this.scmodel{scle}.Basis(:,inds(b)))));
                    b = b + 1;
                end
            end

            if ~isempty(saveName)
                if scle == 1
                    tag = '_coarse';
                elseif scle == 2
                    tag = '_fine';
                end
                saveas(h, strcat(this.savePath, 'selectedBasis', saveName, tag), 'png');
            end
        end

        %% Display 'nSamples' sample patches with the basis used for encoding
        function h = displayEncoding(this, scle, texture, disparity, blurLeft, blurRight, contrastLeft, contrastRight, nSamples, saveName)
            s = sqrt(this.scmodel{scle}.Basis_size/2);  % single patch size
            sBox = 2;                                   % size of black box surrounding the basis
            suppPostNorm = 0;                           % normalization after patch adaptation?
            blurScale = 0.8;

            params={s, this.scmodel{scle}.Dsratio, blurScale, this.cataract_l, this.cataract_r};
            img = imread(sprintf('./textures/%d.bmp', texture)); %example image

            % generate patches (zero mean, unit norm)
            [patchesLeft, patchesRight] = PatchGenerator(img, disparity, blurLeft-this.spectacles_l, blurRight-this.spectacles_r, this, params);

            % contrast adaptation
            newPatchesLeft = patchesLeft .* contrastLeft;
            newPatchesRight = patchesRight .* contrastRight;

            if suppPostNorm
                newPatchesLeft = newPatchesLeft - repmat(mean(newPatchesLeft),[size(newPatchesLeft,1) 1]);
                newPatchesRight = newPatchesRight - repmat(mean(newPatchesRight),[size(newPatchesRight,1) 1]);
            end
            
            [coefs, err, monocularity] = this.scmodel{scle}.suppressiveEncode([newPatchesLeft; newPatchesRight]);

            h = figure;
            hold on;
            rng(1)
            for p = 1 : nSamples

                pIndex = randi(size(patchesLeft, 2)); % take one random patch
                
                % display original patch
                patch = [patchesLeft(:, pIndex), patchesRight(:, pIndex)];
                subplot(nSamples*2, 10, [1 + (20*(p-1)), 11 + (20*(p-1))]);
                imshow(mat2gray(col2im(patch, [s s], [2*s s], 'distinct')))
                title('orig');

                % display altered patch
                patchAdap = [newPatchesLeft(:, pIndex), newPatchesRight(:, pIndex)];
                subplot(nSamples*2, 10, [2 + (20*(p-1)), 12 + (20*(p-1))]);
                imshow(mat2gray(col2im(patchAdap, [s s], [2*s s], 'distinct')))
                title('adapt');

                % display reconstruction
                encPatch = this.scmodel{scle}.Basis*coefs(:, pIndex);
                recIm = [encPatch(1 : end/2), encPatch(end/2+1 : end)];
                subplot(nSamples*2, 10, [3 + (20*(p-1)), 13 + (20*(p-1))]);
                imshow(mat2gray(col2im(recIm, [s s], [2*s s], 'distinct')))
                title('rec.');

                % display selected basis
                binIndx = calculateRightBinocularity(this.scmodel{scle}.Basis);
                corrs = calculateCorrelation(this.scmodel{scle}.Basis);
                [~, bInds]=sort(abs(coefs(:,pIndex)));
                bInds = flip(bInds(end-this.scmodel{scle}.Basis_num_used+1: end));
                for b = 1 : this.scmodel{scle}.Basis_num_used
                    base = this.scmodel{scle}.Basis(:, bInds(b));
                    baseIm = [base(1 : end/2), base(end/2 + 1 : end)];
                    if b <= (this.scmodel{scle}.Basis_num_used / 2)
                        subplot(nSamples*2, 10, (20*(p-1)) + 3 + b);
                    else
                        subplot(nSamples*2, 10, (20*(p-1)) + 8 + b);
                    end
                    imshow(mat2gray(col2im(baseIm, [s s], [2*s s], 'distinct')))
                    text(1,s+1,sprintf('act:%0.2f\nbin:%0.2f\ncorr:%0.2f', coefs(bInds(b), pIndex), binIndx(1, bInds(b)), corrs(1, bInds(b))), 'color', 'y', 'fontsize', 6);
                end

                % display distribution of binocularity
                contribs = binIndx(coefs(:, pIndex) ~= 0);
                sp = subplot(nSamples*2, 10, [(20*(p-1)) + 9, (20*(p-1)) + 10, (20*(p-1)) + 19, (20*(p-1)) + 20]);
                histogram(contribs, linspace(0,1,10));
                title(sprintf('mean BI: %0.2f', mean(contribs)));
                set(sp, 'fontsize', 7);
            end

            i = regexp(this.savePath, '/');
            expName = this.savePath(i(end-1)+1:end-1);  % set the experiment name and params as title above all
            suptitle(sprintf('Exp: %s\ndisp=%1d, blurL=%d, blurR=%d, contrL=%.1f, contrR=%.1f',...
                strrep(expName, '_', '\_'), disparity, blurLeft-this.spectacles_l, blurRight-this.spectacles_r, contrastLeft, contrastRight));

            if ~isempty(saveName)
                if scle == 1
                    tag = '_coarse';
                elseif scle == 2
                    tag = '_fine';
                end
                saveas(h, sprintf('%s/patchEncoding%s_%s.png', this.savePath, tag, saveName));
            end
        end

        %% Displays the correlation of activated basis-sub-parts over multiple disparities
        function displayCorrelation(this, scle, saveTag)

            disparityRange = [-3:1:3];
            blurLeft = 0;
            blurRight = 0;

            s = sqrt(this.scmodel{scle}.Basis_size/2);    % single patch size
            sBox = 2;                               % size of black box surrounding the basis

            params={s, this.scmodel{scle}.Dsratio, 0.625, this.cataract_l, this.cataract_r};
            img = imread(sprintf('/home/klimmasch/projects/AccomodationLearning_orig/textures_mcgill/%d.bmp', img)); %example image

            nBins = 100;
            histograms = zeros(nBins, length(disparityRange));

            for d = 1 : length(disparityRange)

                [patchesLeft, patchesRight] = PatchGenerator(img, disparityRange(d), blurLeft, blurRight, this, params);
                % imshow(mat2gray(col2im(patchesLeft, [10 10], [110 110], 'distinct')))
                [coefs, err, monocularity] = this.scmodel{scle}.suppressiveEncode([patchesLeft; patchesRight]);
                pIndex = 1;
                binIndx = calculateBinocularity(this.scmodel{scle}.Basis);
                corrs = calculateCorrelation(this.scmodel{scle}.Basis);
                [~, bInds]=sort(abs(coefs(:,pIndex)));
                bInds = flip(bInds(end-this.scmodel{scle}.Basis_num_used+1: end, :)); %for each patch, 10 indices of selected basis
                h = histogram(corrs(bInds(:)), nBins);
                histograms(:, d) = h.Values;
            end

            f = figure;
            hold on;
            surf(histograms);
            set(gca, 'XTickLabel', [-3:1:3]);
            set(gca, 'YTickLabel', [-1:0.4:1]);
            xlabel('Disparity');
            ylabel('Freq. of Correlation');
            view([24, 30]);

            if ~isempty(saveTag)
                if scle == 1
                    tag = '_coarse';
                elseif scle == 2
                    tag = '_fine';
                end
                saveas(h, sprintf('%s/correlation_scale$d_%s', scle, this.savePath, saveTag),'png');
            end
        end

        function displayBinHist(this, scle, nBins, saveName)
            binocs = calculateRightBinocularity(this.scmodel{scle}.Basis);
            h = figure;
            histogram(binocs,linspace(0,1,nBins));
            xlabel('Binocularity Index');
            ylabel('Frequency');
            title(sprintf('mean BI = %0.2f', mean(binocs)));
            if ~isempty(saveName)
                if scle == 1
                    tag = '_coarse';
                elseif scle == 2
                    tag = '_fine';
                end
                saveas(h, strcat(this.savePath, saveName, tag,'.png'));
            end
        end

        %% calculates the reward surface for a given set of basis functions over different blur levels
        % textures specifies an array of images over which the reward is averaged.
        function rewards = rewardSurface(this, scle, basisAt, textures, displ, saveName)
            blurScale = 0.8;

            contrasts = [0, 0.5, 1, 1.5, 2];

            contrastsR = flip(contrasts);
            blurLevels = -3:3;

            rewards = zeros(length(blurLevels), length(contrasts));   % reward coarse, fine
            % rewards = zeros(length(blurLevels), length(textures));  % for this version, show reward surface for each img
                                                                      % over blur levels and contrasts

            for s = 1:length(this.scmodel)
                  for b = 1:length(blurLevels)
                      for texture = 1:length(textures)
                            img = imread(sprintf('%s%d.bmp', this.texture_directory, textures(texture)));

                            for c = 1:length(contrasts)
                                  contrastLeft = contrasts(c);
                                  contrastRight = contrastsR(c);
                                  [~, ~, blurRewLeft, blurRewRight] = PatchGenerator(img, 0, blurLevels(b)+this.spectacles_l, blurLevels(b)+this.spectacles_r, this, {10, this.scmodel{scle}.Dsratio, blurScale, this.cataract_l, this.cataract_r});

                                  blurRewLeft = blurRewLeft * contrastLeft;
                                  blurRewRight = blurRewRight * contrastRight;

                                  rew = blurRewLeft + blurRewRight;
                                  rewards(b, c) = rew + rewards(b, c);
                            end
                      end
                  end
            end

            rewards = rewards ./ length(textures);

            if displ
                h = figure('Position', [500 800 600 600]);
                surf(rewards(:, :));
                axis tight;

                s1 = gca();
                s1.XTick = 1:length(contrasts);
                s1.XTickLabel = contrasts;
                s1.YTick = 1:length(blurLevels);
                s1.YTickLabel = blurLevels;

                xlabel('contrast left');
                ylabel('blur error left');
                title('coarse scale');

                if ~isempty(saveName)
                    if scale == 1
                        tag = '_coarse';
                    elseif scale == 2
                        tag = '_fine';
                    end
                    saveas(gca, sprintf('%srewardSurf_%sScale_%s.png', this.savePath, tag, saveName));
                end
            end
        end

        %% Plots results from one simulation run
        function allPlotSave(this)
            
            % predefining some colors
            orange=[255 153 24]./255;
            blue=[39 48 73]./255;
            green=[146 208 80]./255;
            lightblue=[70 166 199]./255;
            purple=[139 84 166]./255;
            brown=[167 121 29]./255;
            gray=[100 100 100]/255;

            for s = 1:size(this.scmodel,2)
                this.displayBasis(s, 20, 15, [], 1); % display and save basis
                try
                    this.displaySelectedBasis(s, 4, 4, '_');
                catch
                    sprintf('It seems no basis where selected that could be displayed.')
                end

                this.displayEncoding(s, 66, 0, 0, 0, this.contrast_hist(this.trainedUntil, 1), this.contrast_hist(this.trainedUntil, 2), 2, 'example');
            end

            if this.useSuppression
                h = figure;
                windowsize = ceil(this.T_train/100);
                conts = filter(ones(1, windowsize) / windowsize, 1, this.contrast_hist);%(windowsize+1:end));
                plot(conts);
                ylim([0, 2]);
                title('Left and Right Contrasts over Train Time');
                xlabel('Time');
                ylabel('Contrasts');
                legend('left eye, coarse', 'right eye, coarse', 'left eye, fine', 'right eye, fine');
                saveas(gca, strcat(this.savePath, 'conts_hist.png'));
                close(h);
            end


            %% blur level
            windowSize = ceil(this.T_train/500);
            ind = 1:this.T_train;
            i = mod(ind,this.Interval);
            ind = find(i); %to exclude trials where vergence is reset
            ind=ind((this.Interval-1):(this.Interval-1):end);

            blur_r = filter(ones(1,windowSize)/windowSize,1,abs(this.reward_hist(ind,2)+this.spectacles_r));
            blur_r = blur_r(windowSize:end); %discard first values (mean not reliable)
            blur_l = filter(ones(1,windowSize)/windowSize,1,abs(this.reward_hist(ind,2)+this.spectacles_l));
            blur_l = blur_l(windowSize:end); %discard first values (mean not reliable)
            ind = ind(windowSize:end);

            h=figure; hold on;
            plot(ind,blur_r,'-','LineWidth',1,'Color',green); xlabel('Time','FontSize',8); ylabel('Stdv [px]','FontSize',8);
            plot(ind,blur_l,'-','LineWidth',1,'Color',lightblue); xlabel('Time','FontSize',8); ylabel('Stdv [px]','FontSize',8);
            ylim([0, Inf])
            title('Moving Average of Blur Level');
            set(gca,'XTick',0:this.T_train/5:this.T_train,'FontSize',8); grid on

            saveas(gca, strcat(this.savePath, 'blur level'), 'png');
            close(h);

            %% mean VergError
            windowSize = ceil(this.T_train/500);
            ind = 1:this.T_train;
            i = mod(ind,this.Interval);
            ind = find(i); %to exclude trials where vergence is resetted
            ind=ind((this.Interval-1):(this.Interval-1):end);

            vergerr = filter(ones(1,windowSize)/windowSize,1,abs(this.reward_hist(ind,1)));
            vergerr = vergerr(windowSize:end); %discard first values (mean not reliable)
            ind = ind(windowSize:end);
            h = figure; hold on;
            plot(ind,vergerr,'--','LineWidth',1,'Color',blue);
            set(gca,'FontName','LM Roman 8');
            ylim([0 Inf]);
            set(gca,'LineWidth',1);
            set(gca,'XTick',0:this.T_train/2.5:this.T_train,'FontSize',8);
            set(gca,'YTick',0:1:3,'FontSize',8);
            
            title('Moving average of errors during training','FontSize',8,'FontWeight','Normal');

            xlabel('Time','FontSize',8);
            ylabel('Error [px]','FontSize',8);

            saveas(gca, strcat(this.savePath, 'mean VergError'), 'png');
            close(h);

            %% recostruction error
            windowSize = ceil(this.T_train/100);
            recerr = filter(ones(1,windowSize)/windowSize,1,this.recerr_hist);
            h = figure; hold on
            plot(recerr,'k','LineWidth',2);
            xlabel('Time','FontSize',14);
            ylabel('Rec Error [AU]','FontSize',14);
            title('Moving Average of Reconstruction Error');
            set(gca,'XTick',0:this.T_train/10:this.T_train,'FontSize',14); grid on;

            saveas(gca, strcat(this.savePath, 'rec_error'), 'png');
            close(h);

            %% acc and verg error after last step of sequence at the beginning and the end of learning:
            h = figure; hold on;
            if this.T_train >= 1000
                plot(this.vergerr_hist(this.Interval-1+end-1000:this.Interval:end),'b.-','LineWidth',1,'MarkerSize',12);%start
            else
                plot(this.vergerr_hist(this.Interval-1:this.Interval:end),'b.-','LineWidth',1,'MarkerSize',12);
            end
            xlabel('Time','FontSize',14);
            ylabel('VergError [px]','FontSize',14);
            title('VergErr for 250 consecutive iterations','FontSize',10);
            set(gca,'XTick',0:10:250,'FontSize',6); grid on;

            saveas(gca, strcat(this.savePath, 'last 250 last-step-vergErrs'), 'png');
            close(h);

            %% last accerror after last step in sequence
            h = figure; hold on;
            if this.T_train >= 1000
                plot(this.accerr_hist(this.Interval-1+end-1000:this.Interval:end),'b.-','LineWidth',1,'MarkerSize',12);
            else
                plot(this.accerr_hist(this.Interval-1:this.Interval:end),'b.-','LineWidth',1,'MarkerSize',12);
            end
            xlabel('Trial Number','FontSize',14);
            ylabel('AccError [px]','FontSize',14);
            title('AccErr for 250 consecutive iterations','FontSize',10);
            set(gca,'XTick',0:10:250,'FontSize',6); grid on;

            saveas(gca, strcat(this.savePath, 'last 250 last-step-AccErrs'), 'png');
            close(h);

            %% show how acc and verg actions interact
            h = figure; hold on;
            plot(this.accerr_hist(end-100+1:1:end),'.-','LineWidth',1,'MarkerSize',12,'Color',lightblue);
            plot(this.vergerr_hist(end-100+1:1:end),'.-','LineWidth',1,'MarkerSize',12,'Color',blue);
            xlabel('Iteration','FontSize',12);
            ylabel('Error [px]','FontSize',12);
            ylim([-6.5 6.5]);
            title('Vergence and accommodation error in 100 consecutive iterations','FontSize',8);
            legend('acc. error left','verg. error','Location', 'North');grid on;
            set(gca,'XTick',0:10:100,'FontSize',12);
            set(gca,'YGrid','off');

            saveas(gca, strcat(this.savePath, 'verg_and_acc_err'), 'png');
            close(h);

            %% plot accommodation and vergence plane trajectories
            if ~isempty(this.accPlane_hist)
                interv=100;

                h = figure; hold on;
                plot(this.objPlane_hist(this.trainedUntil-interv+1:this.trainedUntil),'--','LineWidth',1,'MarkerSize',12,'Color',gray);
                plot(this.accPlane_hist(this.trainedUntil-interv+1:this.trainedUntil)+this.spectacles_r,'.-','LineWidth',1,'MarkerSize',12,'Color',green);
                plot(this.accPlane_hist(this.trainedUntil-interv+1:this.trainedUntil)+this.spectacles_l,'.-','LineWidth',1,'MarkerSize',12,'Color',lightblue);
                plot(this.vergPlane_hist(this.trainedUntil-interv+1:this.trainedUntil),'.-','LineWidth',1,'MarkerSize',12,'Color',blue);
                xlabel('Iteration','FontSize',12);
                ylabel('Plane Position','FontSize',12);
                % ylim([-this.maxObjPlane-1 this.maxObjPlane+1]);
                title(['Plane positions in ' num2str(interv) ' consecutive iterations - spectacles(l,r):(' num2str(+this.spectacles_l) ',' num2str(+this.spectacles_r) ')'],'FontSize',8);
                legend('Obj. Plane', 'Acc. Plane Right','Acc. Plane Left','Verg. Plane','Location', 'North');grid on;
                set(gca,'XTick',0:10:interv,'FontSize',12);
                set(gca,'YGrid','off');

                saveas(gca, strcat(this.savePath, 'vergErrs-AccErr interaction3'), 'png');
                close(h);
            end

            %% mean errors in each iteration in sequence
            for i=1:this.Interval
                mean_iter_acc_err(i)=mean(abs(this.accerr_hist(this.Interval-1+i:this.Interval:end)));
                mean_iter_verg_err(i)=mean(abs(this.vergerr_hist(this.Interval-1+i:this.Interval:end)));
                sigm_iter_acc_err(i)=std(abs(this.accerr_hist(this.Interval-1+i:this.Interval:end)));
                sigm_iter_verg_err(i)=std(abs(this.vergerr_hist(this.Interval-1+i:this.Interval:end)));
            end

            h = figure; hold on;
            errorbar(1:this.Interval,mean_iter_acc_err,sigm_iter_acc_err);
            errorbar(1:this.Interval,mean_iter_verg_err,sigm_iter_verg_err);
            xlabel('Iteration within Trial','FontSize',14);
            ylabel('Error [px]','FontSize',14);
            legend('Acc. Error','Verg. Error');

            saveas(gca, strcat(this.savePath, 'errors per iteration'), 'png');
            close(h);

            %% error and action path for each step in sequence (also for specific textures)
            err_hi=this.accerr_hist+normrnd(0,0.1,size(this.accerr_hist,1),size(this.accerr_hist,2));
            err_hi=err_hi(1:end-1);

            h = figure; hold on;
            for idx=2:min(1000+2, (size(err_hi,1)+1)/this.Interval-1)
                x_data = 1:this.Interval;
                y_data = err_hi( floor(((size(err_hi,1)+1)/this.Interval-idx) * this.Interval) : floor((((size(err_hi,1)+1)/this.Interval-idx+1)) * this.Interval-1));
                h=plot(x_data,y_data,'Color',blue);
                c = 100*1/min(1000, size(err_hi,1)/this.Interval-2);  % 50% transparent
                if c > 1
                    c = 1;
                elseif c < 0
                    c = 0;
                end

                h.Color(4) = c;
            end

            grid on;
            title('Left Acc. Error Trajectories (scattered)','FontSize',12);
            xlabel('Iteration within Trial','FontSize',12);
            ylabel('Acc. Error Left','FontSize',12);
            xlim([1 this.Interval]);
            ylim([min(err_hi) max(err_hi)]);

            saveas(gca, strcat(this.savePath, 'error and action path for each iteration (also textures)'), 'png');

            %% heatmap of errors during one fixation
            interv = this.trainedUntil/2;
            accerrs = this.objPlane_hist(this.trainedUntil - interv + 1: this.trainedUntil - 1) - (this.accPlane_hist(this.trainedUntil - interv + 1: this.trainedUntil - 1) + this.spectacles_l);
            stepInSeq = repmat(1:this.Interval, 1, interv/10);
            stepInSeq = stepInSeq(2:end);

            blurInd = - min(accerrs);
            maxBlur = max(accerrs) + blurInd;

            pairs = [stepInSeq', accerrs, this.objPlane_hist(this.trainedUntil-interv+1:this.trainedUntil-1), this.accPlane_hist(this.trainedUntil-interv+1:this.trainedUntil-1), this.reward_hist(this.trainedUntil-interv+1:this.trainedUntil-1,3)];
            mapping = zeros(this.Interval, maxBlur + 1);

            for i = 1:length(pairs)
                pair = pairs(i, :);
                mapping(pair(1), pair(2) + blurInd + 1) = mapping(pair(1), pair(2) + blurInd + 1) + 1;
            end
            mapping = mapping';
            
            h = figure;
            imagesc(1:this.Interval, min(accerrs):max(accerrs), mapping);
            colormap('gray');
            title('Left Acc. Error');
            xlabel('Iteration within Trial','FontSize',12);
            ylabel('acc error','FontSize',12);
            set(gca,'XTick',2:2:this.Interval,'FontSize',12);
            set(gca,'YTick',min(accerrs):2:max(accerrs),'FontSize',12);

            saveas(gca, strcat(this.savePath, 'acc errs 2nd half of training'), 'png');
            close(h);

            %% average errors per for each input texture:
            % each dot corresponds to one input image
            ind=this.Interval-1:this.Interval:this.trainedUntil;

            mean_pic_vergerror=accumarray(this.texture_hist(ind),abs(this.vergerr_hist(ind)),[],@(x) mean(x,1));        % accumulates the mean error for each image

            mean_pic_accerror_l=accumarray(this.texture_hist(ind),abs(this.accerr_hist(ind)+this.spectacles_l),[],@(x) mean(x,1));
            mean_pic_accerror_r=accumarray(this.texture_hist(ind),abs(this.accerr_hist(ind)+this.spectacles_r),[],@(x) mean(x,1));

            std_pic_vergerror=accumarray(this.texture_hist(ind),abs(this.vergerr_hist(ind)),[],@(x) std(x,1));

            std_pic_accerror_l=accumarray(this.texture_hist(ind),abs(this.accerr_hist(ind)+this.spectacles_l),[],@(x) std(x,1));
            std_pic_accerror_r=accumarray(this.texture_hist(ind),abs(this.accerr_hist(ind)+this.spectacles_r),[],@(x) std(x,1));

            h = figure; hold on;
            scatter(mean_pic_vergerror,std_pic_vergerror,10,blue);
            scatter(mean_pic_accerror_r,std_pic_accerror_r,10,green);
            scatter(mean_pic_accerror_l,std_pic_accerror_l,10,lightblue);
            title(sprintf('Verg-/Acc-Err for %d input images - LensRight:%dpx', max(this.texture_hist), this.spectacles_r,'FontSize',12));
            ylim([0 2.2]);
            xlim([0 4]);
            set(gca,'XTick',0:1:4,'FontSize',12);
            set(gca,'YTick',0:0.5:2,'FontSize',12);
            xlabel('Mean Error [px]','FontSize',12);
            ylabel('Stdv [px]','FontSize',12);
            legend('Verg. Error','Acc. Error Right','Acc. Error Left','Location', 'NorthEast');
            grid on;

            saveas(gca, strcat(this.savePath, 'error per image'), 'png');
            close(h);

            %% Relation between object distance and left acc error
            objt = this.distances(1:end-1);
            objt = objt';
            acc_err = this.accerr_hist;
            acc_err_seq=acc_err(this.Interval-1:this.Interval:end);  % acc err in last step of sequence

            distIndTransf = - min(objt) + 1;                         % transformation of raw values to indices
            accIndTransf = - min(acc_err_seq) + 1;                   % same for accomodation plane

            %plot heatmap which errors occures with what frequency given the object position
           
            buffer=[objt+distIndTransf acc_err_seq+accIndTransf];
            buffer=accumarray(buffer(:,[1 2]), 1);
            buffer=buffer';
            h = figure;
            colormap('gray');   % set colormap
            imagesc(min(objt):max(objt),min(acc_err_seq)+this.spectacles_l:max(acc_err_seq)+this.spectacles_l,buffer); % draw image and scale colormap to values range
            colorbar;           % show color scale
            title('distribution of acc error in last iteration of sequences for given object distances','FontSize',8);
            xlabel('Object Position','FontSize',12);
            ylabel('Acc. Error Left','FontSize',12);

            saveas(gca, strcat(this.savePath, 'object position vs final AccErr'), 'png');
            close(h);
            
            close all;
        end
    end
end
