function [names, colors] = semanticClasses()
%SEMANTICCLASSES Load the class order and colors shared with Python.
root = fileparts(fileparts(mfilename('fullpath')));
entries = jsondecode(fileread(fullfile(root, 'semantic', 'camvid_classes.json')));
names = string({entries.name});
colors = double(reshape([entries.color], 3, []).') ./ 255;
end
