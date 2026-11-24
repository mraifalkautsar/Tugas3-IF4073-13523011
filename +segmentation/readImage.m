function rgb = readImage(path)
%READIMAGE Read the first frame of a grayscale, indexed, or RGB image.
[image, palette] = imread(path);
if ~isempty(palette)
    image = ind2rgb(image, palette);
end
rgb = segmentation.asRGB(image);
end
