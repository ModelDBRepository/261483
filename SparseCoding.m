classdef SparseCoding < handle
    properties
        Basis_num_used;   % number of basis used to encode in sparse mode
        Basis_size;       % size of each base vector
        Basis_num;        % total basis number
        Basis;            % all the basis
        Basis_hist;       % save basis at regular intervals
        Basis_selected;   % indicates for each basis how often it has been selected
        eta;              % learning rate
        Temperature;      % temperature in softmax
        Dsratio;          % downsampling ratio (to produce 8x8)
        patch_size;       % size of extracted patches
    end
    methods
        %PARAM = {Basis_num_used,Basis_size,Basis_num,eta,Temperature,Dsratio,Basis_S};
        function obj = SparseCoding(PARAM, nSaves)
            obj.Basis_num_used = PARAM{1};
            obj.Basis_size = PARAM{2};
            obj.Basis_num = PARAM{3};
            obj.Basis_selected = zeros(PARAM{3}, 1);
            obj.eta = PARAM{4};
            obj.Temperature = PARAM{5};
            obj.Dsratio = PARAM{6};
            obj.patch_size = PARAM{8};

            % initialize receptive field as white noise
            a=rand(obj.Basis_size,obj.Basis_num)-0.5; 
            a=a*diag(1./sqrt(sum(a.*a)));
            thenorm = ones(obj.Basis_size,1)*sqrt(sum(a.*a,1));
            a=a./thenorm;
            obj.Basis = a;
            obj.Basis_hist = zeros(obj.Basis_size, obj.Basis_num, nSaves);
            obj.Basis_hist(:, :, 1) = a;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%
        %%% encode the image accoring to softmax distribution
        %%%
        %%% Images is the batch input
        %%% debugmode indicates whether some intermedia should be recorded;
        %%%
        %%% Coef is the output Coefficients for each basis and images
        %%% Error is the reconstruction error using current coefficients
        %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [Coef, Error] = softmaxEncode(this,Images)
            batch_size = size(Images,2);
            Coef = zeros(this.Basis_num,batch_size);
            I = Images;
            for count = 1:this.Basis_num_used
                corr = abs(this.Basis'*I)/this.Temperature;
                corr = corr - kron(ones(this.Basis_num,1),max(corr));
                softmaxcorr = softmax(corr);
                
                softmaxcorr = tril(ones(this.Basis_num))*softmaxcorr - repmat(rand(1,batch_size),[this.Basis_num 1]); %faster than 'kron'
                softmaxcorr(softmaxcorr<0) = 2;
                [~,index] = min(softmaxcorr);
                corr = this.Basis'*I;
                linearindex = sub2ind(size(corr),index,1:batch_size);
                Coef(linearindex) = Coef(linearindex) + corr(linearindex);
                I = Images - this.Basis*Coef;
            end
            Error = I;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%
        %%% Encode the input images with the best matched basis
        %%%
        %%% Images are the input images batch
        %%%
        %%% Coef is the output Coefficients
        %%% Error is the reconstruction error using current coefficients
        %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [coef, error, monocularity] = sparseEncode(this,imageBatch)
            size_Batch = size(imageBatch,2);
            coef = zeros(this.Basis_num,size_Batch);
            imageOrig = imageBatch;
            corr = this.Basis'*imageBatch;      %correlation of each basis with each patch
            corrBB = this.Basis'*this.Basis;    %correlation between basis
            for count = 1:this.Basis_num_used

                [~,index] = max(abs(corr));                             % indices of bases with max corr per patch
                linearindex = sub2ind(size(corr),index,1:size_Batch);   % corresponding linear indices in corr matrix
                pCorr = corr(linearindex);                              % vector of correlations per patch (coefs per patch)
                coef(linearindex) = coef(linearindex) + pCorr;          % stores corr coefs into coef matrix

                corr = corr - bsxfun(@times,corrBB(:,index),pCorr);
            end
            error = imageOrig - this.Basis*coef;

            usedBasis = zeros(size(coef));
            usedBasis(find(coef)) = 1;
            usedBasis = sum(usedBasis, 2);
            this.Basis_selected = usedBasis;
            
            feature = mean(coef.^2, 2);
            feature = feature ./ sum(feature);
            
            binInds = calculateRightBinocularity(this.Basis);  % returns right monocular dominance 
            weightedMonocs = binInds' .* feature;
            monocularity = sum(weightedMonocs);
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%
        %%% Calculate the correlation between input image and the basis
        %%%
        %%% Images are the input image batch
        %%%
        %%% Coef is the output correlation
        %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [Coef,Error] = fullEncode(this,Images)
            Coef = this.Basis'*Images;
            Error = Images - this.Basis*Coef;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%
        %%% Update the basis
        %%%
        %%% Coef is the input coefficient
        %%% Error is the input error
        %%% debugmode indicates whether some intermedia should be recorded;
        %%%
        %%% Basis_Change is the changing amount of the basis in current
        %%% update
        %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function updateBasis(this,coef,error)
            deltaBases = error * coef'/size(error,2);
            this.Basis = this.Basis + this.eta*deltaBases;
            this.Basis = bsxfun(@rdivide,this.Basis,sqrt(sum(this.Basis.^2)));
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%
        %%% train sparse coding for one step
        %%%
        %%% Images is the input image batch
        %%% debugmode indicates whether some intermedia should be recorded;
        %%%
        %%% Error is the reconstruction error using the best matched coefficients
        %%% Basis_picked indicates which basis are picked to encode
        %%% Basis_Entropy is the entropy of each base
        %%% Basis_Change is the changing amount of the basis in current
        %%% update
        %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [error, coef, monocularity] = stepTrain(this, Images)
            [coef, error, monocularity] = this.sparseEncode(Images);    % matching pursuit
            updateBasis(this, coef, error);                             % adapt RFs via gradient descent
        end

        function [error, coef, monocularity, weightedMonocs] = suppressiveStepTrain(this, Images)
            [coef, error, monocularity, weightedMonocs] = this.suppressiveEncode(Images);
            updateBasis(this, coef, error);                                 
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%
        %%% save the parameters in a file
        %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function saveClass(this, configfile)
            Basis = this.Basis;
            save(configfile, 'Basis', '-append');
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%
        %%% save the Basis during training
        %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function saveBasis(this, index)
            % this.Basis_hist = cat(3, this.Basis_hist, this.Basis);
            this.Basis_hist(:, :, index) = this.Basis;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%
        %%% Encode the input images with the best matched basis
        %%% and indicate the use of monocular basis functions
        %%%
        %%% imageBatch - preprocessed batch of input patches
        %%%
        %%% Coef - correlation coeffizients
        %%% Error - reconstruction error
        %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [coef, error, monocularity, weightedMonocs] = suppressiveEncode(this,imageBatch)
            size_Batch = size(imageBatch,2);
            coef = zeros(this.Basis_num,size_Batch);
            imageOrig = imageBatch;
            corr = this.Basis'*imageBatch;      % correlation of each basis with each patch
            corrBB = this.Basis'*this.Basis;    % correlation between basis
            for count = 1:this.Basis_num_used

                [~,index] = max(abs(corr));                             % indices of bases with max corr per patch
                linearindex = sub2ind(size(corr),index,1:size_Batch);   % corresponding linear indices in corr matrix
                pCorr = corr(linearindex);                              % vector of correlations per patch (coefs per patch)
                coef(linearindex) = coef(linearindex) + pCorr;          % stores corr coefs into coef matrix
                corr = corr - bsxfun(@times,corrBB(:,index),pCorr);     
            end
            error = imageOrig - this.Basis * coef;

            usedBasis = zeros(size(coef));
            usedBasis(find(coef)) = 1;
            usedBasis = sum(usedBasis, 2);
            this.Basis_selected = usedBasis;
            
            feature = mean(coef.^2, 2);
            feature = feature ./ sum(feature);
            binInds = calculateRightBinocularity(this.Basis);
            
            weightedMonocs = binInds' .* feature;
            monocularity = sum(weightedMonocs);
        end
        
        function [coef, error, monocularity, weightedMonocs] = suppressiveEncodeAt(this, basisAt, imageBatch)
            size_Batch = size(imageBatch,2);
            coef = zeros(this.Basis_num,size_Batch);
            imageOrig = imageBatch;
            basis = this.Basis_hist(:, :, basisAt);

            corr = basis' * imageBatch;      % correlation of each basis with each patch
            corrBB = basis' * basis;         % correlation between basis
            
            for count = 1:this.Basis_num_used
                [~,index] = max(abs(corr));                             % indices of bases with max corr per patch
                linearindex = sub2ind(size(corr),index,1:size_Batch);   % corresponding linear indices in corr matrix
                pCorr = corr(linearindex);                              % vector of correlations per patch (coefs per patch)
                coef(linearindex) = coef(linearindex) + pCorr;          % stores corr coefs into coef matrix

                corr = corr - bsxfun(@times,corrBB(:,index),pCorr);
            end
            error = imageOrig - basis * coef;

            usedBasis = zeros(size(coef));
            usedBasis(find(coef)) = 1;
            usedBasis = sum(usedBasis, 2);
            this.Basis_selected = usedBasis;
            
            feature = mean(coef.^2, 2);
            feature = feature ./ sum(feature);
            binInds = calculateRightBinocularity(basis); 
            
            weightedMonocs = binInds' .* feature;
            monocularity = sum(weightedMonocs);
        end
        
    end
end
