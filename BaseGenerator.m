
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% Generates set of receptive fields with specific frequency
%%% and orientation distribution.
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function basis = BaseGenerator(saveBasis, displayBasis, base_size, base_num)

    s=base_size;
    n=base_num;
    
    sigma=2.5;       % sigma of gaussian envelop of gabor patch
    a=[];
    for i=1:n
       
        orient=rand()*180;                     % orientation measured from horizontal axis anticlockwise in degrees
        theta=rand()*360;                      % phase shift of sin wave in degrees
        lambda = rand()*s*5/3+s*1/3;           % wavelength of sin wave in px - uniform lambda distribution results in 1/f^2 freq. distribution
        shift_x=rand()*s*2/3-s*1/3;            % shift wavelet away from patch center
        shift_y=rand()*s*2/3-s*1/3;
        delta_orient=2;
        while(abs(delta_orient)>=1)
            delta_orient=laprnd(1, 1, 0, 0.5);
        end
        delta_orient=delta_orient*90;
        
        dispa=2;
        while(abs(dispa)>=1)
            dispa=laprnd(1, 1, 0, 1);
        end
        dispa=dispa*s/8;
         
        p=0.35;                 % set balance of eye dominance. positive values equal left monocular bias
        corr=2;                 % sets balance between left and right eye receptive subfield., -1=right monocular, 1=left monocular
        while(abs(corr)>=1)
            corr=normrnd(p,0.4);
        end
                
        % generate receptive field
        base_l = Gabor(sigma,orient,lambda,theta,lambda/(0.8*s),s,[shift_x shift_y],+dispa);
        base_r = Gabor(sigma,orient+delta_orient,lambda,theta,lambda/(0.8*s),s,[shift_x shift_y],-dispa);

        %normalize left and right part of base
        base_l=base_l(:)-mean(base_l(:));              % zero mean
        base_r=base_r(:)-mean(base_r(:));
        base_l=base_l/sqrt(sum(base_l.^2));            % normalize to unit norm
        base_r=base_r/sqrt(sum(base_r.^2));
        
        base=[(1+corr)*base_l(:);(1-corr)*base_r(:)];  % the more dissimilar left and right patch are the more monocular the RF becomes
        a=[a, base];                                   % add receptive field to collection
    end

    basis=a;

    basis=bsxfun(@rdivide,basis,sqrt(sum(basis.^2)));  % normalize to unit norm, joint left and right subfield

    if saveBasis
        save('basis.mat','basis');
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% Display Receptive Fields that have been generated
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if displayBasis
        
        %####################################
        orange=[255 153 24]./255;
        blue=[39 48 73]./255;
        green=[146 208 80]./255;
        lightblue=[70 166 199]./255;
        purple=[139 84 166]./255;
        brown=[167 121 29]./255;
        gray=[80 80 80]/255;
        bgcolor= [255 255 255]./255;
        %####################################
        
        basisDisp = basis(:,:);

        s=sqrt(size(basisDisp,1)/2);
        n=size(basisDisp,2);

        block=mat2gray(col2im(basisDisp,[s 2*s],[n*s 2*s],'distinct'));

        M=10;  % number of bases displayed in one row
        N=5;     % maximum number of rows to plot

        k=1;
        pic=[];

        for j=1:N
            row=[];
            for i=1:M
                base_l=block(1+(k-1)*s:k*s,1:s);
                base_r=block(1+(k-1)*s:k*s,s+1:2*s);
                brick=[-ones(1,s); base_l; -ones(1,s); base_r; -ones(1,s)]; %left receptive field part on top, right receptive field at bottom
                brick=[-ones(2*s+3,1) brick -ones(2*s+3,1)];
                row=[row brick];
                k=k+1;
            end

            pic=[pic;row];
        end

        % %add frame
        % pic=[-ones(size(pic,1),1) pic -ones(size(pic,1),1)];
        % pic=[-ones(1,size(pic,2)); pic; -ones(1,size(pic,2))];

        imshow(pic);

        %colorize
        red = pic;
        green = pic;
        blue = pic;

        red(red==-1)= bgcolor(1);
        green(green==-1)= bgcolor(2);
        blue(blue==-1)= bgcolor(3);

        test = zeros(size(pic,1),size(pic,2),3);

        test(:,:,1)=red;
        test(:,:,2)=green;
        test(:,:,3)=blue;

        imshow(test)
    end
