
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% Calculate the correlation between the left and the right subfield
%%% of receptive fields.
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function correlation = calculateCorrelation(Bases)

LeftEye = Bases(1:end/2,:);
RightEye = Bases(end/2+1:end,:);
correlation = dot(LeftEye, RightEye) ./ (sqrt(sum(LeftEye.^2)) .* sqrt(sum(RightEye.^2)));

end