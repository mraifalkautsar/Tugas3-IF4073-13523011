function smokeSegmentationApp(checkpoint)
%SMOKESEGMENTATIONAPP Exercise the actual Run callback in each MATLAB tab.
arguments
    checkpoint (1,1) string = ""
end
root = fileparts(fileparts(mfilename('fullpath')));
if strlength(checkpoint) == 0
    imagePath = string(tempname) + ".png";
    shape = zeros(65, 67, 'uint8');
    shape(17:49, 19:50) = 255;
    imwrite(shape, imagePath);
    imageCleanup = onCleanup(@() delete(imagePath)); %#ok<NASGU>
    checkpoint = string(tempname) + ".pt";
    expectPrediction = false;
else
    imagePath = fullfile(root, 'data', 'CamVid', 'CamVid_RGB', '0001TP_008550.png');
    expectPrediction = true;
end
window = segmentationApp(InitialImage=imagePath, Checkpoint=checkpoint, Device="cpu");
windowCleanup = onCleanup(@() delete(window)); %#ok<NASGU>
tabs = findall(window, 'Tag', 'Methods');
runButton = findall(window, 'Tag', 'Run');
status = findall(window, 'Tag', 'Status');

tabs.SelectedTab = findall(window, 'Tag', 'Edge');
runButton.ButtonPushedFcn(runButton, []);
assert(contains(status.Text, 'foreground pixels'), status.Text);
tabs.SelectedTab = findall(window, 'Tag', 'Split');
runButton.ButtonPushedFcn(runButton, []);
assert(contains(status.Text, 'final regions'), status.Text);
tabs.SelectedTab = findall(window, 'Tag', 'Semantic');
runButton.ButtonPushedFcn(runButton, []);
if expectPrediction
    assert(contains(status.Text, 'classes detected'), status.Text);
else
    assert(strcmp(status.Text, 'Processing failed.'), status.Text);
end
assert(strcmp(runButton.Enable, 'on'));
disp('MATLAB GUI callbacks completed.');
end
