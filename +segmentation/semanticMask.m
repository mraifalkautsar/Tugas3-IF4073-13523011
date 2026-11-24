function result = semanticMask(image, options)
%SEMANTICMASK Run the CamVid model in WSL and validate the returned mask.
arguments
    image
    options.Distro (1,1) string = "Ubuntu"
    options.Python (1,1) string = ".venv/bin/python"
    options.Checkpoint (1,1) string = ""
    options.Device (1,1) string {mustBeMember(options.Device, ["auto", "cpu", "cuda"])} = "auto"
end

root = fileparts(fileparts(mfilename('fullpath')));
checkpoint = options.Checkpoint;
if strlength(checkpoint) == 0
    checkpoint = fullfile(root, 'artifacts', 'best.pt');
end
if ~isfile(checkpoint)
    error('segmentation:MissingCheckpoint', ...
        'Trained CamVid checkpoint is missing. Train the model or install best.pt at %s.', checkpoint);
end
if ~ispc || ~usejava('jvm')
    error('segmentation:MissingRuntime', 'CamVid inference requires Windows MATLAB with Java and Ubuntu WSL.');
end

wsl = fullfile(getenv('SystemRoot'), 'System32', 'wsl.exe');
if ~isfile(wsl)
    error('segmentation:MissingRuntime', 'Install WSL and the Ubuntu distribution for CamVid inference.');
end

rgb = segmentation.asRGB(image);
job = tempname;
mkdir(job);
cleanup = onCleanup(@() rmdir(job, 's')); %#ok<NASGU>
input = fullfile(job, 'image.png');
output = fullfile(job, 'classes.png');
imwrite(im2uint8(rgb), input);
base = [string(wsl), '-d', options.Distro];
linuxRoot = linuxPath(root);
linuxJob = linuxPath(job);
linuxCheckpoint = linuxPath(checkpoint);

command = [base, '--cd', linuxRoot, '--exec', options.Python, ...
    '-m', 'semantic', 'predict', '--input', linuxJob + "/image.png", ...
    '--output', linuxJob + "/classes.png", '--checkpoint', linuxCheckpoint, ...
    '--device', options.Device];
runProcess(command, 180, fullfile(job, 'prediction.log'));
if ~isfile(output)
    error('segmentation:MissingPrediction', 'The CamVid process did not return a mask.');
end

[labels, palette] = imread(output);
[names, colors] = segmentation.semanticClasses();
if ~isempty(palette) || ~isa(labels, 'uint8') || ...
        ~isequal(size(labels), [size(rgb, 1), size(rgb, 2)]) || any(labels(:) >= numel(names))
    error('segmentation:InvalidPrediction', 'The CamVid process returned an invalid class mask.');
end
colorMask = reshape(colors(double(labels(:)) + 1, :), size(rgb));
result = struct('labels', labels, 'colorMask', colorMask, 'overlay', 0.55*rgb + 0.45*colorMask);

    function converted = linuxPath(path)
        file = java.io.File(char(path));
        absolute = string(char(file.getAbsolutePath()));
        converted = runProcess([base, '--cd', '/', '--exec', '/usr/bin/wslpath', '-u', absolute], ...
            30, fullfile(job, 'wslpath.log'));
    end
end

function output = runProcess(command, timeout, logPath)
% Pass each argument directly to WSL so file names never go through a shell.
args = java.util.ArrayList();
for arg = command
    args.add(java.lang.String(char(arg)));
end
builder = java.lang.ProcessBuilder(args);
builder.redirectErrorStream(true);
builder.redirectOutput(java.io.File(logPath));
try
    process = builder.start();
catch exception
    error('segmentation:MissingRuntime', 'Cannot start Ubuntu WSL. %s', exception.message);
end
seconds = javaMethod('valueOf', 'java.util.concurrent.TimeUnit', 'SECONDS');
if ~process.waitFor(int64(timeout), seconds)
    process.destroyForcibly();
    process.waitFor(int64(5), seconds);
    error('segmentation:PredictionTimeout', 'The CamVid process timed out after %d seconds.', timeout);
end
output = strtrim(string(fileread(logPath)));
if process.exitValue() ~= 0
    error('segmentation:PredictionFailed', ['CamVid could not run in Ubuntu WSL. ' ...
        'Check the WSL distribution, Python environment, and checkpoint. %s'], output);
end
end
