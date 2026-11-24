function rgb = asRGB(image)
%ASRGB Convert a grayscale or RGB image to normalized double RGB.
validateattributes(image, {'numeric', 'logical'}, {'real', 'nonempty', 'finite'});
if ismatrix(image)
    image = repmat(image, 1, 1, 3);
elseif ndims(image) ~= 3 || size(image, 3) ~= 3
    error('segmentation:InvalidImage', 'The image must be grayscale or RGB.');
end
rgb = im2double(image);
if any(rgb(:) < 0 | rgb(:) > 1)
    error('segmentation:InvalidImage', 'Normalized image values must be in [0, 1].');
end
end
