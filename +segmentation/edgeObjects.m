function result = edgeObjects(image, operator, options)
%EDGEOBJECTS Detect edges and extract the largest qualifying enclosed region.
arguments
    image
    operator (1,1) string {mustBeMember(operator, ...
        ["Laplace", "LoG", "Sobel", "Prewitt", "Roberts", "Canny"])}
    options.Threshold double = []
    options.Sigma (1,1) double {mustBeInRange(options.Sigma, 0.3, 10)} = 1.4
    options.CloseRadius (1,1) double {mustBeInteger, mustBeNonnegative} = 3
    options.MinAreaFraction (1,1) double {mustBeInRange(options.MinAreaFraction, 0, 1)} = 0.005
end

rgb = segmentation.asRGB(image);
gray = im2gray(rgb);
threshold = options.Threshold;
if ~isempty(threshold)
    validateattributes(threshold, {'double'}, {'scalar', 'real', 'finite', '>=', 0, '<=', 1});
elseif operator ~= "Canny"
    threshold = 0.15;
end

switch operator
    case "Laplace"
        kernel = [0 1 0; 1 -4 1; 0 1 0];
        edgeMap = zeroCrossings(imfilter(gray, kernel, 'replicate', 'conv'), threshold);
    case "LoG"
        radius = ceil(3 * options.Sigma);
        [x, y] = meshgrid(-radius:radius);
        radiusSquared = x.^2 + y.^2;
        sigmaSquared = options.Sigma^2;
        kernel = (radiusSquared - 2*sigmaSquared) .* exp(-radiusSquared/(2*sigmaSquared));
        % Subtract the DC component introduced by truncating the continuous LoG.
        kernel = kernel - mean(kernel(:));
        edgeMap = zeroCrossings(imfilter(gray, kernel, 'replicate', 'conv'), threshold);
    case "Sobel"
        horizontal = [-1 0 1; -2 0 2; -1 0 1];
        edgeMap = gradientEdges(gray, horizontal, horizontal.', threshold);
    case "Prewitt"
        horizontal = [-1 0 1; -1 0 1; -1 0 1];
        edgeMap = gradientEdges(gray, horizontal, horizontal.', threshold);
    case "Roberts"
        edgeMap = gradientEdges(gray, [1 0; 0 -1], [0 1; -1 0], threshold);
    case "Canny"
        if max(gray(:)) == min(gray(:))
            edgeMap = false(size(gray));
        else
            edgeMap = edge(gray, 'Canny', threshold, options.Sigma);
        end
end

closed = edgeMap;
if options.CloseRadius > 0
    closed = imclose(closed, strel('disk', options.CloseRadius, 0));
end

% Subtract the edge bands to obtain just the regions enclosed by contours.
interiors = imfill(closed, 4, 'holes') & ~closed;
components = bwconncomp(interiors, 4);
mask = false(size(gray));
if components.NumObjects > 0
    [area, largest] = max(cellfun(@numel, components.PixelIdxList));
    if area >= options.MinAreaFraction * numel(gray)
        mask(components.PixelIdxList{largest}) = true;
    end
end

result = struct('edgeMap', edgeMap, 'mask', mask, 'cutout', rgb .* mask, ...
    'found', any(mask(:)), 'threshold', threshold);
end

function map = gradientEdges(gray, kernelX, kernelY, threshold)
gradientX = imfilter(gray, kernelX, 'replicate', 'conv');
gradientY = imfilter(gray, kernelY, 'replicate', 'conv');
magnitude = hypot(gradientX, gradientY);
peak = max(magnitude(:));
map = magnitude > max(eps, threshold * peak);
end

function map = zeroCrossings(response, threshold)
localLow = imerode(response, true(3));
localHigh = imdilate(response, true(3));
strength = localHigh - localLow;
% A sign change with low contrast is likely to be a noise edge.
map = localLow < 0 & localHigh > 0 & strength > max(32*eps, threshold * max(strength(:)));
end
