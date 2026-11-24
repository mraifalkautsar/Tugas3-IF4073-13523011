function [edgeRuns, splitRuns, sweeps] = evaluateBSDS500(dataset, output, presetFile)
%EVALUATEBSDS500 Record all operators and split-and-merge settings on the assignment images.
arguments
    dataset (1,1) string = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data', 'bsds500_test_8')
    output (1,1) string = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'results', 'bsds500')
    presetFile (1,1) string = fullfile(fileparts(mfilename('fullpath')), 'bsds500_presets.csv')
end

operators = ["Laplace", "LoG", "Sobel", "Prewitt", "Roberts", "Canny"];
files = dir(fullfile(dataset, '**', '*'));
files = files(~[files.isdir]);
filenames = string({files.name});
isImage = endsWith(lower(filenames), [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]);
files = files(isImage & ~startsWith(filenames, '.'));
[~, order] = sort(string({files.name}));
files = files(order);
if numel(files) ~= 8
    error('segmentation:BSDS500Count', ...
        'Expected the eight assignment images in %s, found %d.', dataset, numel(files));
end

presets = table();
if isfile(presetFile)
    presets = readtable(presetFile, 'TextType', 'string');
end
edgeFolder = fullfile(output, 'edge');
splitFolder = fullfile(output, 'split');
sweepFolder = fullfile(output, 'sweep');
for folder = [edgeFolder, splitFolder, sweepFolder]
    if ~isfolder(folder)
        mkdir(folder);
    end
end
edgeRows = cell(numel(files) * numel(operators), 8);
splitRows = cell(numel(files), 6);
sweepRows = cell(0, 5);
row = 0;

for index = 1:numel(files)
    file = files(index);
    [~, stem] = fileparts(file.name);
    rgb = segmentation.readImage(fullfile(file.folder, file.name));
    original = im2uint8(rgb);
    edgeSheet = cell(numel(operators), 1);
    for op = 1:numel(operators)
        operator = operators(op);
        radius = 3;
        sigma = 1.4;
        minimum = 0.005;
        threshold = 0.15;
        if operator == "Canny"
            threshold = [];
        end
        if ~isempty(presets)
            match = presets.Image == file.name & presets.Operator == operator;
            if any(match)
                setting = presets(find(match, 1), :);
                radius = setting.CloseRadius;
                sigma = setting.Sigma;
                minimum = setting.MinAreaFraction;
                threshold = setting.Threshold;
                if isnan(threshold)
                    threshold = [];
                end
            end
        end
        result = segmentation.edgeObjects(rgb, operator, Threshold=threshold, ...
            Sigma=sigma, CloseRadius=radius, MinAreaFraction=minimum);
        edgeSheet{op} = [original, repmat(uint8(result.edgeMap) * 255, 1, 1, 3), im2uint8(result.cutout)];
        row = row + 1;
        used = result.threshold;
        if isempty(used)
            used = NaN;
        end
        edgeRows(row, :) = {string(file.name), operator, used, sigma, radius, ...
            minimum, result.found, nnz(result.mask)};
    end
    imwrite(vertcat(edgeSheet{:}), fullfile(edgeFolder, stem + ".png"));

    baseline = segmentation.splitMerge(rgb);
    imwrite([original, im2uint8(baseline.image)], fullfile(splitFolder, stem + ".png"));
    splitRows(index, :) = {string(file.name), 0.08, 0.10, 8, ...
        baseline.splitCount, baseline.regionCount};
    if ismember(index, [1 ceil(numel(files)/2)])
        settings = [0.04 0.10; 0.14 0.10; 0.08 0.05; 0.08 0.16];
        comparison = cell(4, 1);
        for setting = 1:size(settings, 1)
            s = settings(setting, 1);
            m = settings(setting, 2);
            variant = segmentation.splitMerge(rgb, SplitThreshold=s, MergeThreshold=m);
            comparison{setting} = im2uint8(variant.image);
            sweepRows(end+1, :) = {string(file.name), s, m, variant.splitCount, variant.regionCount}; %#ok<AGROW>
        end
        imwrite([comparison{1}, comparison{2}; comparison{3}, comparison{4}], ...
            fullfile(sweepFolder, stem + ".png"));
    end
    fprintf('%s: %d/%d edge masks, %d split regions, %d merged regions\n', ...
        file.name, sum(cellfun(@(value) logical(value), edgeRows(row-5:row, 7))), ...
        numel(operators), baseline.splitCount, baseline.regionCount);
end

edgeRuns = cell2table(edgeRows, 'VariableNames', ...
    {'Image', 'Operator', 'Threshold', 'Sigma', 'CloseRadius', 'MinAreaFraction', 'Found', 'AreaPx'});
splitRuns = cell2table(splitRows, 'VariableNames', ...
    {'Image', 'SplitThreshold', 'MergeThreshold', 'MinChildSide', 'InitialRegions', 'FinalRegions'});
sweeps = cell2table(sweepRows, 'VariableNames', ...
    {'Image', 'SplitThreshold', 'MergeThreshold', 'InitialRegions', 'FinalRegions'});
writetable(edgeRuns, fullfile(output, 'edge_parameters.csv'));
writetable(splitRuns, fullfile(output, 'split_parameters.csv'));
writetable(sweeps, fullfile(output, 'split_sweeps.csv'));
end
