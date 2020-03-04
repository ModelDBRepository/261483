
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% Calculates the right monocular dominance of receptive fields
%%% 0: left monocular, 0.5: binocular, 1: right monocular
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function binocularity = calculateRightBinocularity(Bases)
    LeftEye = Bases(1:end/2,:);
    RightEye = Bases(end/2+1:end,:);
    binocularity = (sqrt(sum(RightEye.^2)))./(sqrt(sum(LeftEye.^2)) + sqrt(sum(RightEye.^2)));  % monocular dominance index
    binocularity = squeeze(binocularity(1,:));
end