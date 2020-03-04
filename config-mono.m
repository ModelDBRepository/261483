function model = config(savePath)

%% Sparse/Efficient Coding 
%###############################################################################################################
%parameters in brackets represent those used for multiple scales

Basis_num_used = [10, 10];                      % number of basis used to encode in sparse mode
Basis_size = 8.^2*2;                            % size of each (binocular) base vector
Basis_num= [300,300];                           % total basis number
eta = 0.00005; % 0.05;                                     % learning rate for base adaptation
Temperature = 0.01;                             % temperature in softmax encoding [not used @tm]
Dsratio = [4, 1];                               % the distance of n pixel equals 'n/Dsratio' Pixel after downsampling. Usual values: 1,2,4,8
n_scales = length(Basis_num);                   % number of scales
patch_size = [8, 8];                            % size of the extracted patches (patch_size x patch_size)
%-------------
for s = 1:n_scales
    PARAMSC{s} = {Basis_num_used(s), Basis_size, Basis_num(s), eta, Temperature, Dsratio(s), n_scales, patch_size(s)};
end


%% Reinforcement Learning
%###############################################################################################################
action = 1:5;                                   % number of actions for RL agents
alpha_v = 0.02;                                 % learning rate to update the value network(0.05)
alpha_n = 0.01;                                 % learning rate of natural policy gradient (0.05)
alpha_p = 0.008;                                % learning rate to update the policy network (1)
xi = 0.6;                                       % td discount factor
gamma = 0.01;                                   % learning rate to update cumulative value/ mean reward; (0.01)
Temperature = 1;                                % temperature in softmax function in policy network
S0 = sum(Basis_num);                            % number of neurons in the input layer
weight_range = [0.35,0.05];                     % maximum initial weight
lambda = 0.04;                                  % reguralization factor - additional weight decay term
Act_blur = [-2 -1 0 1 2];                       % action spaces of acc RL agent
Act_disp = [-2 -1 0 1 2];                       % action spaces of verg RL agent
%-------------
PARAMRL_BLUR = {action,alpha_v,alpha_n,alpha_p,xi,gamma,Temperature,lambda,S0,weight_range,Act_blur};
PARAMRL_DISP = {action,alpha_v,alpha_n,alpha_p,xi,gamma,Temperature,lambda,S0,weight_range,Act_disp};


%% setup parameters for model
%###############################################################################################################
trainTime = 5*10^6;                              % total number of interations in the training, in total. Usually 5 000 000
Interval = 10;                                   % Number of iterations in one fixation, i.e., before input texture is repositioned
saveAt = 0 : ceil(trainTime / 20) : trainTime;   % 20+1 datapoints during training where everything is saved

spectacles_l = -2;                                % lense in front of left eye which leads lense to focus at different position plane when relaxed
spectacles_r = 2;                                % lense in front of right eye which leads lense to focus at different position plane when relaxed
cataract_l = 0;                                  % additional blur in left eye. 0: healthy condition, >0: stdv (in px) of additional gaussian blur on left eye
cataract_r = 0;                                  % additional blur in right eye. 0: healthy condition, >0: stdv (in px) of additional gaussian blur on right eye

useSuppression = 1;                              % toggles suppression on or off
threshold = 0.6;                                 % activation threshold of contrast units
saturation = 0.8;                                % saturation of contrast units
slope = saturation/(1-threshold);                % slope of non-linearity of contrast units
eta_supp = 0.1;                                  % learning rate for exponential runing average
exci = 1;                                        % switches excitatory effect of suppression model on or off
suppr = 1;                                       % switches suppresive effect of suppression model on or off

SUPARAMS = {useSuppression, threshold, slope, eta_supp, exci, suppr};

% more general simulation-related parameters
trainRL = 1;                                     % turns reinforcement learning on or off
trainSC = 1;                                     % turns sparse coding on or off
n_textures = 300;                                % number of textures used as inputs
normFeat = 1;                                    % indicates if the feature vector should be normalized

max_obj_plane = 4;                               % target plane positioned between +-max_obj_plane
max_acc_disp_plane = 2;                          % acc and disp plane between +- max_acc_disp_plane
max_disp = max_obj_plane + max_obj_plane;        % largest absolute possible disparity value
max_blur = max_obj_plane + max_acc_disp_plane;   % largest possible blur value

% parameters related to input rendering
texture_directory = './textures/';               % folder that contains the input images
image_size = 300;                                % size of input images
colored_input_images = 0;                        % use gray values or rgb
window_size = 280;                               % size of initial cut out window before disparity is applied 
blur_scale = 0.8;                                % set relation between 1 [a.u.] of accommodation error and stdv of defocus blur filter
alpha = 10^-3;                                   % update rate for running averages of reward normalization

PARAMModel = {0,trainTime,Interval, ...
              spectacles_l,spectacles_r,cataract_l,cataract_r,savePath,trainRL,max_disp,max_blur,saveAt, ...
              SUPARAMS,window_size,alpha,blur_scale,trainSC,n_textures, ...
              texture_directory,image_size,colored_input_images,max_obj_plane,max_acc_disp_plane};

%% ###############################################################################################################

PARAM = {PARAMModel,PARAMSC,PARAMRL_BLUR,PARAMRL_DISP};
model = Model(PARAM);   %create model instance
