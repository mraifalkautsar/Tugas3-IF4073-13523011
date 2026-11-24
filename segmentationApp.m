function window = segmentationApp(options)
%SEGMENTATIONAPP Open the IF4073 segmentation GUI.
arguments
    options.InitialImage (1,1) string = ""
    options.Distro (1,1) string = "Ubuntu"
    options.Python (1,1) string = ".venv/bin/python"
    options.Checkpoint (1,1) string = ""
    options.Device (1,1) string {mustBeMember(options.Device, ["auto", "cpu", "cuda"])} = "auto"
end

if ~license('test', 'image_toolbox') || isempty(which('edge'))
    error('segmentation:MissingToolbox', 'Install and license the MATLAB Image Processing Toolbox.');
end

rgb = [];
running = false;
window = uifigure('Name', 'IF4073 · Image Segmentation', ...
    'Position', [100 100 1150 650], 'Visible', 'off');
main = uigridlayout(window, [3 1], 'RowHeight', {38, '1x', 24});
header = uigridlayout(main, [1 3], 'ColumnWidth', {128, '1x', 100}, 'Padding', [0 0 0 0]);
openButton = uibutton(header, 'Text', 'Open image…', 'ButtonPushedFcn', @browse);
fileLabel = uilabel(header, 'Text', 'No image loaded', 'Interpreter', 'none');
runButton = uibutton(header, 'Text', 'Run', 'Enable', 'off', 'Tag', 'Run', 'ButtonPushedFcn', @run);
tabs = uitabgroup(main, 'Tag', 'Methods');
edgePage = uitab(tabs, 'Title', 'Edge detection', 'Tag', 'Edge');
splitPage = uitab(tabs, 'Title', 'Split and merge', 'Tag', 'Split');
semanticPage = uitab(tabs, 'Title', 'CamVid semantic', 'Tag', 'Semantic');
status = uilabel(main, 'Text', 'Open an image to begin.', 'Tag', 'Status');

edgeGrid = uigridlayout(edgePage, [2 3], 'RowHeight', {42, '1x'});
edgeSettings = uigridlayout(edgeGrid, [1 10], ...
    'ColumnWidth', {53, 96, 66, 56, 56, 52, 56, 47, 78, 55}, 'Padding', [0 0 0 0]);
edgeSettings.Layout.Column = [1 3];
uilabel(edgeSettings, 'Text', 'Operator');
operator = uidropdown(edgeSettings, 'Items', {'Laplace', 'LoG', 'Sobel', 'Prewitt', 'Roberts', 'Canny'}, ...
    'Value', 'Canny', 'ValueChangedFcn', @edgeControls);
automatic = uicheckbox(edgeSettings, 'Text', 'Auto', 'Value', true, 'ValueChangedFcn', @edgeControls, ...
    'Tooltip', 'Estimate the Canny thresholds from the image');
threshold = uieditfield(edgeSettings, 'numeric', 'Limits', [0 1], 'Value', 0.15, ...
    'Tooltip', 'Threshold as a fraction of the peak response, or the high threshold for Canny');
uilabel(edgeSettings, 'Text', 'Sigma');
sigma = uieditfield(edgeSettings, 'numeric', 'Limits', [0.3 10], 'Value', 1.4);
uilabel(edgeSettings, 'Text', 'Closing');
closing = uieditfield(edgeSettings, 'numeric', 'Limits', [0 100], ...
    'RoundFractionalValues', 'on', 'Value', 3, 'Tooltip', 'Closing radius in pixels');
uilabel(edgeSettings, 'Text', 'Min area %');
area = uieditfield(edgeSettings, 'numeric', 'Limits', [0 100], 'Value', 0.5);
edgeOriginal = imageAxes(edgeGrid, 2, 1, 'Original');
edgeOutput = imageAxes(edgeGrid, 2, 2, 'Edge map');
cutoutOutput = imageAxes(edgeGrid, 2, 3, 'Object cutout');
edgeControls();

splitGrid = uigridlayout(splitPage, [2 2], 'RowHeight', {42, '1x'});
splitSettings = uigridlayout(splitGrid, [1 6], ...
    'ColumnWidth', {93, 60, 102, 60, 113, 60}, 'Padding', [0 0 0 0]);
splitSettings.Layout.Column = [1 2];
uilabel(splitSettings, 'Text', 'Split threshold');
splitThreshold = uieditfield(splitSettings, 'numeric', 'Limits', [0 0.5], 'Value', 0.08);
uilabel(splitSettings, 'Text', 'Merge threshold');
mergeThreshold = uieditfield(splitSettings, 'numeric', 'Limits', [0 0.5], 'Value', 0.10);
uilabel(splitSettings, 'Text', 'Min child side (px)');
minSide = uieditfield(splitSettings, 'numeric', 'Limits', [1 Inf], 'RoundFractionalValues', 'on', 'Value', 8);
splitOriginal = imageAxes(splitGrid, 2, 1, 'Original');
splitOutput = imageAxes(splitGrid, 2, 2, 'Split and merge');

semanticGrid = uigridlayout(semanticPage, [2 3], ...
    'RowHeight', {42, '1x'}, 'ColumnWidth', {'1x', '1x', 134});
semanticSettings = uigridlayout(semanticGrid, [1 2], ...
    'ColumnWidth', {50, 118}, 'Padding', [0 0 0 0]);
semanticSettings.Layout.Column = [1 3];
uilabel(semanticSettings, 'Text', 'Device');
device = uidropdown(semanticSettings, 'Items', {'Auto', 'CPU', 'CUDA'}, ...
    'ItemsData', {'auto', 'cpu', 'cuda'}, 'Value', char(options.Device));
semanticOriginal = imageAxes(semanticGrid, 2, 1, 'Original');
semanticOutput = imageAxes(semanticGrid, 2, 2, 'CamVid overlay');
[names, colors] = segmentation.semanticClasses();
legend = uigridlayout(semanticGrid, [numel(names)+1 2], ...
    'ColumnWidth', {14, '1x'}, 'RowHeight', [repmat({22}, 1, numel(names)), {'1x'}], ...
    'Padding', [3 8 0 0], 'RowSpacing', 4, 'ColumnSpacing', 6);
legend.Layout.Row = 2;
legend.Layout.Column = 3;
for index = 1:numel(names)
    uilabel(legend, 'BackgroundColor', colors(index, :), 'Text', '');
    uilabel(legend, 'Text', names(index));
end

if strlength(options.InitialImage) > 0
    loadImage(options.InitialImage);
end
window.Visible = 'on';

    function edgeControls(~, ~)
        isCanny = strcmp(operator.Value, 'Canny');
        automatic.Visible = matlab.lang.OnOffSwitchState(isCanny);
        threshold.Enable = matlab.lang.OnOffSwitchState(~(isCanny && automatic.Value));
        sigma.Enable = matlab.lang.OnOffSwitchState(isCanny || strcmp(operator.Value, 'LoG'));
    end

    function browse(~, ~)
        [file, folder] = uigetfile({'*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff;*.gif', 'Image files'}, ...
            'Open an image');
        if isequal(file, 0)
            return
        end
        try
            loadImage(fullfile(folder, file));
        catch exception
            uialert(window, exception.message, 'Cannot open image');
        end
    end

    function loadImage(path)
        rgb = segmentation.readImage(path);
        [~, name, extension] = fileparts(path);
        fileLabel.Text = string(name) + extension;
        showImage(edgeOriginal, rgb, 'Original');
        showImage(splitOriginal, rgb, 'Original');
        showImage(semanticOriginal, rgb, 'Original');
        cla(edgeOutput);
        cla(cutoutOutput);
        cla(splitOutput);
        cla(semanticOutput);
        status.Text = 'Choose a method and run it.';
        runButton.Enable = 'on';
    end

    function run(~, ~)
        if running || isempty(rgb)
            return
        end
        running = true;
        runButton.Enable = 'off';
        openButton.Enable = 'off';
        reenable = onCleanup(@ready); %#ok<NASGU>
        selected = tabs.SelectedTab;
        status.Text = 'Processing image…';
        drawnow limitrate
        try
            switch selected.Tag
                case 'Edge'
                    value = threshold.Value;
                    if strcmp(operator.Value, 'Canny') && automatic.Value
                        value = [];
                    end
                    result = segmentation.edgeObjects(rgb, operator.Value, Threshold=value, ...
                        Sigma=sigma.Value, CloseRadius=closing.Value, MinAreaFraction=area.Value / 100);
                    showImage(edgeOutput, result.edgeMap, 'Edge map');
                    showImage(cutoutOutput, result.cutout, 'Object cutout');
                    if result.found
                        status.Text = sprintf('%s · %d foreground pixels', operator.Value, nnz(result.mask));
                    else
                        status.Text = 'No enclosed object. Adjust the threshold, closing radius, or minimum area.';
                    end
                case 'Split'
                    result = segmentation.splitMerge(rgb, SplitThreshold=splitThreshold.Value, ...
                        MergeThreshold=mergeThreshold.Value, MinChildSide=minSide.Value);
                    showImage(splitOutput, result.image, 'Split and merge');
                    status.Text = sprintf('%d split regions · %d final regions', result.splitCount, result.regionCount);
                case 'Semantic'
                    cla(semanticOutput);
                    result = segmentation.semanticMask(rgb, Distro=options.Distro, ...
                        Python=options.Python, Checkpoint=options.Checkpoint, Device=device.Value);
                    showImage(semanticOutput, result.overlay, 'CamVid overlay');
                    status.Text = sprintf('%d classes detected in this image', numel(unique(result.labels)));
            end
        catch exception
            if isvalid(window)
                status.Text = 'Processing failed.';
                uialert(window, exception.message, 'Segmentation error');
            end
        end
    end

    function ready()
        running = false;
        if isvalid(window)
            openButton.Enable = 'on';
            runButton.Enable = 'on';
        end
    end
end

function axes = imageAxes(grid, row, column, label)
axes = uiaxes(grid, 'Visible', 'off');
axes.Layout.Row = row;
axes.Layout.Column = column;
title(axes, label, 'Visible', 'on');
end

function showImage(axes, pixels, label)
imshow(pixels, 'Parent', axes);
title(axes, label, 'Visible', 'on');
end
