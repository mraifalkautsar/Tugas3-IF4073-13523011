function result = splitMerge(image, options)
%SPLITMERGE Partition an image by luminance variance and merge adjacent leaves.
arguments
    image
    options.SplitThreshold (1,1) double {mustBeInRange(options.SplitThreshold, 0, 0.5)} = 0.08
    options.MergeThreshold (1,1) double {mustBeInRange(options.MergeThreshold, 0, 0.5)} = 0.10
    options.MinChildSide (1,1) double {mustBeInteger, mustBePositive} = 8
end

rgb = segmentation.asRGB(image);
gray = im2gray(rgb);
[height, width] = size(gray);
labels = zeros(height, width, 'uint32');
pixelSums = integralImage(gray);
squareSums = integralImage(gray.^2);
counts = zeros(0, 1);
sums = zeros(0, 1);
squares = zeros(0, 1);
split(1, height, 1, width);
leafCount = numel(counts);

% Horizontal and vertical changes in the leaf map capture every shared side.
pairs = [neighborPairs(labels(:, 1:end-1), labels(:, 2:end)); ...
    neighborPairs(labels(1:end-1, :), labels(2:end, :))];
pairs = unique(sort(pairs, 2), 'rows');
adjacent = sparse(double(pairs(:, 1)), double(pairs(:, 2)), ...
    ones(size(pairs, 1), 1), leafCount, leafCount);
adjacent = logical(adjacent + adjacent.');
parent = (1:leafCount).';
queue = parent;
queued = true(leafCount, 1);
head = 1;

while head <= numel(queue)
    node = queue(head);
    head = head + 1;
    queued(node) = false;
    if parent(node) ~= node
        continue
    end

    for neighbor = find(adjacent(:, node)).'
        joinedCount = counts(node) + counts(neighbor);
        joinedSum = sums(node) + sums(neighbor);
        joinedSquares = squares(node) + squares(neighbor);
        if deviation(joinedCount, joinedSum, joinedSquares) > options.MergeThreshold
            continue
        end

        keep = min(node, neighbor);
        drop = max(node, neighbor);
        parent(drop) = keep;
        counts(keep) = joinedCount;
        sums(keep) = joinedSum;
        squares(keep) = joinedSquares;

        linked = adjacent(:, keep) | adjacent(:, drop);
        linked([keep drop]) = false;
        adjacent(:, drop) = false;
        adjacent(drop, :) = false;
        adjacent(:, keep) = linked;
        adjacent(keep, :) = linked.';

        % Only the new region and its neighbors have changed merge options.
        affected = [keep; find(linked)];
        pending = affected(~queued(affected));
        queue = [queue; pending]; %#ok<AGROW>
        queued(pending) = true;
        break
    end
end

% Smaller surviving IDs ensure that each parent root is already resolved.
for leaf = 1:leafCount
    parent(leaf) = parent(parent(leaf));
end
[roots, ~, denseIds] = unique(parent);
finalLabels = reshape(uint32(denseIds(double(labels))), height, width);
rendered = zeros(size(rgb));
for channel = 1:3
    plane = rgb(:, :, channel);
    meanColors = accumarray(double(finalLabels(:)), plane(:), ...
        [numel(roots) 1], @mean);
    rendered(:, :, channel) = reshape(meanColors(finalLabels), height, width);
end
result = struct('labels', finalLabels, 'image', rendered, ...
    'splitCount', leafCount, 'regionCount', numel(roots));

    function split(top, bottom, left, right)
        rows = bottom - top + 1;
        cols = right - left + 1;
        count = rows * cols;
        total = rectangleSum(pixelSums, top, bottom, left, right);
        squared = rectangleSum(squareSums, top, bottom, left, right);
        canSplitRows = rows >= 2 * options.MinChildSide;
        canSplitCols = cols >= 2 * options.MinChildSide;

        if ~(canSplitRows || canSplitCols) || ...
                deviation(count, total, squared) <= options.SplitThreshold
            leaf = numel(counts) + 1;
            labels(top:bottom, left:right) = leaf;
            counts(leaf, 1) = count;
            sums(leaf, 1) = total;
            squares(leaf, 1) = squared;
            return
        end

        rowCuts = [top, bottom + 1];
        colCuts = [left, right + 1];
        if canSplitRows
            rowCuts = [top, top + floor(rows / 2), bottom + 1];
        end
        if canSplitCols
            colCuts = [left, left + floor(cols / 2), right + 1];
        end
        for row = 1:numel(rowCuts)-1
            for col = 1:numel(colCuts)-1
                split(rowCuts(row), rowCuts(row+1)-1, colCuts(col), colCuts(col+1)-1);
            end
        end
    end
end

function total = rectangleSum(integral, top, bottom, left, right)
total = integral(bottom+1, right+1) - integral(top, right+1) ...
    - integral(bottom+1, left) + integral(top, left);
end

function sigma = deviation(count, total, squared)
% The pooled moments include differences between the original region means.
variance = squared / count - (total / count)^2;
sigma = sqrt(max(0, variance - 8*eps));
end

function pairs = neighborPairs(first, second)
boundary = first ~= second;
from = first(boundary);
to = second(boundary);
pairs = [from(:), to(:)];
end
