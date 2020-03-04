
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 
%%% This function produces a numerical approximation to 2D Gabor function.
%%% Parameters:
%%% sigma  = standard deviation of Gaussian envelope, this in-turn controls the
%%%          size of the result (pixels)
%%% orient = orientation of the Gabor clockwise from the vertical (degrees)
%%% wavel  = the wavelength of the sin wave (pixels)
%%% phase  = the phase of the sin wave (degrees)
%%% aspect = aspect ratio of Gaussian envelope (0 = no "width" to envelope, 
%%%          1 = circular symmetric envelope)
%%% pxsize = the size of the filter (optional). If not specified, size is 5*sigma.
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function gb=Gabor(sigma,orient,wavel,phase,aspect,pxsize,shift,dispa)

if nargin<6
  pxsize=fix(5*sigma);
end

if mod(pxsize,2)~=0
    [x, y]=meshgrid(-fix(pxsize/2):fix(pxsize/2),-fix(pxsize/2):fix(pxsize/2));
else
    [x, y]=meshgrid(-fix(pxsize/2)+0.5:fix(pxsize/2)-0.5,-fix(pxsize/2)+0.5:fix(pxsize/2)-0.5);
end

x = x+dispa;  % add disparity tuing

orient=-orient*pi/180;  % rotate subfield
x_theta=x*cos(orient)+y*sin(orient);
y_theta=-x*sin(orient)+y*cos(orient);

phase=phase*pi/180;
freq=2*pi./wavel;

shift_x=shift(1);
shift_y=shift(2);

gb=exp(-0.5.*( (((x_theta+shift_x).^2)/(sigma^2)) ...
			 + (((y_theta+shift_y).^2)/((aspect*sigma)^2)) )) ...
   .* (cos(freq*y_theta+phase) - cos(phase).*exp(-0.25.*((sigma*freq).^2)));