function tests = testMatlabSegmentation
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
root = fileparts(fileparts(mfilename('fullpath')));
testCase.applyFixture(matlab.unittest.fixtures.PathFixture(root));
end

function testAllEdgeOperators(testCase)
operators = ["Laplace", "LoG", "Sobel", "Prewitt", "Roberts", "Canny"];
uniform = ones(65, 67) * 0.4;
shape = zeros(65, 67);
shape(17:49, 19:50) = 1;
truth = logical(shape);

for operator = operators
    clean = segmentation.edgeObjects(uniform, operator);
    verifyFalse(testCase, any(clean.edgeMap(:)), operator);
    verifyFalse(testCase, clean.found, operator);
    result = segmentation.edgeObjects(shape, operator);
    verifyTrue(testCase, result.found, operator);
    overlap = nnz(result.mask & truth) / nnz(result.mask | truth);
    verifyGreaterThan(testCase, overlap, 0.65, operator);
    verifySize(testCase, result.cutout, [65 67 3]);
end
end

function testCannyMaximumThreshold(testCase)
shape = zeros(65, 67);
shape(17:49, 19:50) = 1;
result = segmentation.edgeObjects(shape, "Canny", Threshold=1);
verifyEqual(testCase, result.edgeMap, false(size(shape)));
verifyFalse(testCase, result.found);
end

function testThinContoursPreservesEdgeMap(testCase)
shape = zeros(65, 67);
shape(17:49, 19:50) = 1;
standard = segmentation.edgeObjects(shape, "Canny");
thinned = segmentation.edgeObjects(shape, "Canny", ThinContours=true);
verifyEqual(testCase, thinned.edgeMap, standard.edgeMap);
verifyTrue(testCase, thinned.found);
overlap = nnz(thinned.mask & logical(shape)) / nnz(thinned.mask | logical(shape));
verifyGreaterThan(testCase, overlap, 0.65);
end

function testSplitMerge(testCase)
uniform = segmentation.splitMerge(ones(31, 47) * 0.3);
verifyEqual(testCase, uniform.regionCount, 1);
verifyEqual(testCase, uniform.image, repmat(0.3, [31 47 3]), 'AbsTol', 1e-12);

diagonal = [zeros(15, 23), ones(15, 24); ones(16, 23), zeros(16, 24)];
separate = segmentation.splitMerge(diagonal);
verifyEqual(testCase, separate.regionCount, 4);
verifyEqual(testCase, separate.splitCount, 4);
verifySize(testCase, separate.labels, [31 47]);
verifyNotEqual(testCase, separate.labels(1, 1), separate.labels(end, end));
verifyGreaterThan(testCase, min(separate.labels(:)), uint32(0));

horizontal = repmat([zeros(1, 23), ones(1, 24)], 31, 1);
joined = segmentation.splitMerge(horizontal);
verifyEqual(testCase, joined.splitCount, 4);
verifyEqual(testCase, joined.regionCount, 2);
verifyEqual(testCase, joined.labels(1, 1), joined.labels(end, 1));
verifyNotEqual(testCase, joined.labels(1, 1), joined.labels(1, end));

oneRow = segmentation.splitMerge(horizontal(1, :));
verifySize(testCase, oneRow.labels, [1 47]);
verifyEqual(testCase, oneRow.regionCount, 2);
end

function testImageReading(testCase)
directory = tempname;
mkdir(directory);
cleanup = onCleanup(@() rmdir(directory, 's')); %#ok<NASGU>
pixels = uint8([0 1; 1 0]);
path = fullfile(directory, 'indexed image.png');
imwrite(pixels, [1 0 0; 0 1 0], path);
rgb = segmentation.readImage(path);
verifySize(testCase, rgb, [2 2 3]);
verifyEqual(testCase, reshape(rgb(1, 1, :), 1, []), [1 0 0]);

path = fullfile(directory, 'grayscale.jpg');
imwrite(pixels * 255, path);
verifySize(testCase, segmentation.readImage(path), [2 2 3]);
end

function testMissingCheckpoint(testCase)
verifyError(testCase, @() segmentation.semanticMask(zeros(5), ...
    Checkpoint=fullfile(tempdir, 'missing-if4073-checkpoint.pt')), 'segmentation:MissingCheckpoint');
end
