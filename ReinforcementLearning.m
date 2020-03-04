classdef ReinforcementLearning < handle
    properties
        Action;         %actions or command, for example -5:5
        Action_num;     %total number of actions can be taken
        Action_hist;    % action history (Luca) 
        pol_hist;       % policy probabilities history (Luca)
        td_hist;        % history of td errors
        
        alpha_v;        %learning rate to update the value function
        alpha_p;        %learning rate to update the policy function
        alpha_n;        %learning rate to update the nature gradient w
		
        gamma;          %learning rate to update cumulative value;
        lambda;         %the regularizatoin factor
        xi;             %discount factor
        
        Temperature;    %temperature in softmax function in policy network
        weight_range;   %maximum initial weight
        
        S0;             %number of neurons in the input layer
        S1_pol;         %number of neurons in the middle layer of policy network
        S1_val;         %number of neurons in the middle layer of value network
        S2;             %number of actions
        
        Weights;
        
        J = 0;
        g;              %intermedia variable to keep track of "w" which is the gradient of the policy
        X;              %intermedia variable to keep track of last input
        Y_pol;          %intermedia variable to keep track of the values of the middle layer in policy network for last input
        Y_val;          %intermedia variable to keep track of the values of the middle layer in value network for last input
        val;            %intermedia variable to keep track of the value estimated for last input
        pol;            %intermedia variable to keep track of the policy for last input
        label_act;      %intermedia variable to keep track of which action has been chosen
        td;             %intermedia variable to keep track of the TD error    
        Weights_hist;   %weights history

        featMeans;      % mean values of each entry in the feature vector (used for normalization)
        featStds;       % standard deviations of each entry in the feature vector (used for normalization)
        featGamma = 0.1;% exponential weight for updates

    end
    methods
        %PARAM = {Action,alpha_v,alpha_n,alpha_p,xi,gamma,Temperature,lambda,S0,weight_range,loadweights,weights,weights_hist};
        function obj = ReinforcementLearning(PARAM, trainTime, nSaves)
            obj.Action = PARAM{1};
            
            obj.alpha_v = PARAM{2};
			obj.alpha_n = PARAM{3};
			obj.alpha_p = PARAM{4};
			obj.xi = PARAM{5};
            obj.gamma = PARAM{6};
            obj.Temperature = PARAM{7};
			obj.lambda = PARAM{8};
            obj.S0 = PARAM{9};
            obj.S2 = length(obj.Action);
            obj.Action_num = length(obj.Action);
            
            obj.Action_hist = zeros(trainTime, 1);   
            obj.pol_hist = zeros(trainTime, obj.Action_num);      
            obj.td_hist = zeros(trainTime, 1);       
            obj.td = 0;     
            obj.pol = zeros(1, obj.Action_num);
            obj.Weights_hist = cell(2,nSaves);
            
            obj.weight_range = PARAM{10};
             
            obj.featMeans = zeros(PARAM{9}, 1);
            obj.featStds = ones(PARAM{9}, 1);
            
            obj.NAC_initNetwork();
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% initialize the parameters of the class
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function NAC_initNetwork(this)
            this.Weights{1,1} = (2*rand(this.S2,this.S0)-1)*this.weight_range(1); %Actor/Policy
            this.Weights{2,1} = (2*rand(1,this.S0)-1)*this.weight_range(2); %Critic/Value
            
            this.J = 0; %average reward (Eq. 3.6, 3.7) 
            this.g = zeros(this.S2*this.S0,1); %gradient
            this.Weights_hist(:, 1) = this.Weights;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%
        %%% update the parameters in the network
        %%%
        %%% Xin is the input to the network
        %%% reward is the reward for reinforement learning
        %%% flag_update indicates whether the network should updated
        %%%
        %%% D contrains the intermedia values
        %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function D = NAC_updateNetwork(this,Xin,reward,flag_update)
            X_new = Xin; 
            val_new = this.Weights{2,1}*X_new;
            D = zeros(1,32);
            
            if(flag_update) % see Eq. 3.4 to 3.12 in thesis
                this.J = (1-this.gamma) * this.J + this.gamma*reward;
                delta = reward - this.J + this.xi*val_new - this.val;   %TD error
                
                dv_val = delta * this.X';
                this.Weights{2,1} = this.Weights{2,1} + this.alpha_v * dv_val;  %value net/ critic update
                
                dlogv_pol = 1/this.Temperature*(this.label_act-this.pol) * this.X';
                psi = dlogv_pol(:);
                
                %alphaforg = this.alpha; %learning rate for w (g)
                
                deltag = delta * psi - psi*(psi'* this.g);
                this.g = this.g  +  this.alpha_n * deltag;
                
                dlambda = this.g;
                
                % this.td = [this.td delta];    %save TD error
                this.td = delta;
                
                dv_pol = reshape(dlambda(1:numel(dlogv_pol)),size(this.Weights{1,1}));
                
                this.Weights{1,1} = this.Weights{1,1} * (1-this.alpha_p*this.lambda);
                this.Weights{1,1} = this.Weights{1,1} + this.alpha_p*dv_pol;

            end
            this.X = X_new;
            this.val = val_new;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%
        %%% generate command according to the softmax distribution of the
        %%% output in the policy network
        %%% feature is the input to the network
        %%% 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function command = softmaxAct(this,feature)
            Xin = feature;
            
            poltmp = this.Weights{1,1}*Xin/this.Temperature;
            this.pol = softmax(poltmp - max(poltmp));           % the min() will be the result
                        
            % assosiatedsoftmax = this.pol;
            softmaxpol = tril(ones(this.Action_num))*this.pol;
            softmaxpol = softmaxpol/softmaxpol(end)- rand(1);
            
            softmaxpol(softmaxpol<0) = 2;
            [~,index] = min(softmaxpol);
            command = this.Action(index);
            
            this.label_act = zeros(this.Action_num, 1);
            this.label_act(index) = 1;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%
        %%% pick the action with the maximum possibility among all the
        %%% commands calculated in policy network
        %%% feature is the input to the network
        %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [command, pol, feature] = Act(this,feature)            
            poltmp = this.Weights{1,1} * feature / this.Temperature;
            pol = softmax(poltmp - max(poltmp));
            [~, index] = max(poltmp);
            command = this.Action(index);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%
        %%% Train the reinforcement network for one step
        %%%
        %%% feature is the input to the network
        %%% reward is the reward for reinforement learning
        %%% flag_update indicates whether the network should updated
        %%%
        %%% command is the output command
        %%% parameters is the intermedia values keeped for debug
        %%% En is the entropy of policy
        %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        function command = stepTrain(this,feature,reward,flag_update,normFeat)
            if normFeat % normalize feature vector
                for i = 1 : length(feature)
                    feature(i) = this.normFeatVec(feature(i), i, flag_update);
                end
            end
            this.NAC_updateNetwork(feature,reward,flag_update);
            command = this.softmaxAct(feature);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%
        %%% save the weights during training
        %%% 3rd dim corresponds to iteration, col to weight
        %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function saveWeights(this, index)
            this.Weights_hist(1, index) = this.Weights(1); %policy net 
            this.Weights_hist(2, index) = this.Weights(2); %value net
        end
             
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%
        %%% save the parameters in a file
        %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function saveClass(this,configfile)
            weights = cell(2,1);
            weights{1} = this.Weights;
            weights{2} = this.g;
            save(configfile,'weights','-append');
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%
        %%% normalize one entry of the feature vector (not used)
        %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function normedValue = normFeatVec(this, value, index, updateFlag)
            normedValue = (value - this.featMeans(index)) / sqrt(this.featStds(index));
            if updateFlag
                delta = value - this.featMeans(index);
                this.featMeans(index) = this.featMeans(index) + this.featGamma * delta;
                this.featStds(index) = (1 - this.featGamma) * this.featStds(index) + this.featGamma * delta^2;
            end
        end
    end
end