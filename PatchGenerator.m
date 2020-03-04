%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 
%%% Extracts patches from input image.
%%% Applies blur and disparity based on plane positions
%%% Applies whitening filter.
%%% Calculates acc reward. 
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [patchesLeft, patchesRight, rewardLeft, rewardRight] = PatchGenerator(Image, dispa, blur_l, blur_r, model, params)

noise = 0;
whitening = 1;          

%image process vars
patch_size = params{1};  % size of patches
Dsratio = params{2};     % downsampling ratio
blur_scale = params{3};  % plane position difference (between object and acc planes) of 1 leads to gaussian blur stdv of 1*blur_scale
cataract_l = params{4};  % independent of plane position: add blur stdv of cataract_l*blur_scale to left eye
cataract_r = params{5};  % independent of plane position: add blur stdv of cataract_r*blur_scale to right eye

window_size = model.window_size;

s = patch_size./max([Dsratio-2; 1]);   % distance between extracted patches
window = patch_size + s * patch_size;  % in units after downsampling. 'patch_size' is minimal window size, then add additional pixels such that 'patch_size' additional patches can be sampled. Results in 'patch_size+1' many patches per column with slide 's'.
nc = window-patch_size+1;              % maximum number of patches per column (slide of 1 px)

curr_Img=Image;

ImageLeft = imcrop(curr_Img,[(size(curr_Img,1)/2-window_size/2-dispa) (size(curr_Img,2)/2-window_size/2) window_size-1 window_size-1]);
ImageRight = imcrop(curr_Img,[(size(curr_Img,1)/2-window_size/2+dispa) (size(curr_Img,2)/2-window_size/2) window_size-1 window_size-1]);          
		

% defocus blur
if(blur_l~=0)
    h = fspecial('gaussian', ceil(5*abs(blur_l*blur_scale))+mod(ceil(5*abs(blur_l*blur_scale)),2)+1,abs(blur_l*blur_scale));
    ImageLeft = imfilter(ImageLeft,h,'replicate');
end
if(blur_r~=0)
    h = fspecial('gaussian', ceil(5*abs(blur_r*blur_scale))+mod(ceil(5*abs(blur_r*blur_scale)),2)+1,abs(blur_r*blur_scale));
    ImageRight = imfilter(ImageRight,h,'replicate');
end
        
		
imgrawLeft = ImageLeft;
imgrawRight = ImageRight;

% apply cataract to right and left eye
if(cataract_r>0)
    h = fspecial('gaussian', ceil(4*abs(cataract_r*blur_scale)),abs(cataract_r*blur_scale));
    imgrawRight = imfilter(imgrawRight,h,'replicate');
elseif(cataract_r<0)
    imgrawRight=randn(size(imgrawRight,1),size(imgrawRight,2)); % simulate total blindness as random noise;
end

if(cataract_l>0)
    h = fspecial('gaussian', ceil(4*abs(cataract_l*blur_scale)),abs(cataract_l*blur_scale));
    imgrawLeft = imfilter(imgrawLeft,h,'replicate');
elseif(cataract_l<0)
    imgrawLeft=randn(size(imgrawLeft,1),size(imgrawLeft,2)); % simulate total blindness as random noise;
end

if(noise ~= 0)
    imgrawLeft = imnoise(imgrawLeft, 'gaussian', 0, noise);
    imgrawRight = imnoise(imgrawRight, 'gaussian', 0, noise);
end

%% Whitening
%###########################################################################################

if whitening
    
    % parameters for whitening filter
    N=window_size;
    [fx, fy]=meshgrid(-N/2:N/2-1,-N/2:N/2-1);
    rho=sqrt(fx.*fx+fy.*fy);
    f_0=0.4*N;
    phi = 4;
    a = 1.27;
    filt2D = (rho.^a).*exp(-(rho/f_0).^phi);  
            
    %----
    %whitening LEFT eye   
    %----
    curr_Img=imgrawLeft;   
    If=fft2(curr_Img);
    imagew=real(ifft2(If.*fftshift(filt2D)));
    curr_Img=imagew;
    imgrawLeft=curr_Img;
        
    %----
    %whitening RIGHT eye   
    %----
    curr_Img=imgrawRight;
    If=fft2(curr_Img);
    imagew=real(ifft2(If.*fftshift(filt2D)));
    curr_Img=imagew;
    imgrawRight=curr_Img;
    
end


%% Patch extraction left image
%###########################################################################################

img = imgrawLeft;

%Downsample image to 8x8 using Gaussian Pyramid
for i = 1:log2(Dsratio)
    img = impyramid(img,'reduce');
end

img = double(img);

%cut window in the center
[h,w] = size(img);
img = img(fix(h/2+1-window/2):fix(h/2+window/2), fix(w/2+1-window/2):fix(w/2+window/2));

% reward extraction
rewardLeft = mean(mean(img.^2));

%cut patches and store them as col vectors
patches = im2col(img,[patch_size patch_size],'sliding'); %slide window of 1 px

%take patches at steps of s (8 px)
cols = [];
for kc = 1:s:nc
    C = (kc-1)*nc+1:s:kc*nc;
    cols = [cols C];
end

patches = patches(:,cols);
patchesLeft = patches;


%% Patch extraction right image
%###########################################################################################

img = imgrawRight;

%Downsample image to 8x8 using Gaussian Pyramid
for i = 1:log2(Dsratio)
    img = impyramid(img,'reduce');
end

img = double(img);

img = img(fix(h/2+1-window/2):fix(h/2+window/2), fix(w/2+1-window/2):fix(w/2+window/2));

% reward extraction
rewardRight = mean(mean(img.^2));

%cut patches and store them as col vectors
patches = im2col(img,[patch_size patch_size],'sliding'); %slide window of 1 px

%take patches at steps of s (8 px) --> is defined above
%     cols = [];
%     for kc = 1:s:nc
%         C = (kc-1)*nc+1:s:kc*nc;
%         cols = [cols C];
%     end

patches = patches(:, cols);
patchesRight = patches;

%% Normalization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%	

%zero mean
patchesLeft = patchesLeft - repmat(mean(patchesLeft),[size(patchesLeft,1) 1]);
patchesRight = patchesRight - repmat(mean(patchesRight),[size(patchesRight,1) 1]);

% % monocular unit variance
% patchesLeft = bsxfun(@rdivide,patchesLeft,sqrt(sum(patchesLeft.^2))); 
% patchesRight = bsxfun(@rdivide,patchesRight,sqrt(sum(patchesRight.^2))); 

% %joint unit variance
% patches =[patchesLeft;patchesRight];
% patches = bsxfun(@rdivide,patches,sqrt(sum(patches.^2))); 

% %Re-extract image patches of left/right eye
% patchesLeft = patches(1:(size(patches,1)/2),:);
% patchesRight =  patches((size(patches,1)/2)+1:end,:);

end

