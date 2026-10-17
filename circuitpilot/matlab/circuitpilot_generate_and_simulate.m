function resultJson = circuitpilot_generate_and_simulate(specFile, outputDir, resultFile)
% Deterministic Simulink generator for the CircuitPilot control graph IR.
% Uses APIs available in MATLAB/Simulink R2024b.
spec = jsondecode(fileread(specFile));
if ~exist(outputDir, 'dir'), mkdir(outputDir); end
model = char(spec.name);
modelPath = fullfile(outputDir, [model '.slx']);
if bdIsLoaded(model), close_system(model, 0); end
new_system(model);
cleanup = onCleanup(@() close_if_loaded(model)); %#ok<NASGU>
load_system('simulink');

definitions = circuitpilot_component_definitions();
for k = 1:numel(spec.components)
    component = spec.components(k);
    kind = char(component.type);
    blockPath = [model '/' char(component.id)];
    libraryPath = definitions.(matlab.lang.makeValidName(kind));
    if getSimulinkBlockHandle(libraryPath) == -1
        error('CircuitPilot:MissingLibraryBlock', ...
            'Simulink library block does not exist: %s', libraryPath);
    end
    add_block(libraryPath, blockPath, ...
        'Position', [40 + mod(k-1, 5)*150, 40 + floor((k-1)/5)*100, ...
        100 + mod(k-1, 5)*150, 80 + floor((k-1)/5)*100]);
    if isfield(component, 'parameters')
        paramNames = fieldnames(component.parameters);
        for p = 1:numel(paramNames)
            key = paramNames{p};
            value = component.parameters.(key);
            if isnumeric(value), value = mat2str(value); end
            set_param(blockPath, key, char(string(value)));
        end
    end
end

for k = 1:numel(spec.connections)
    connection = spec.connections(k);
    source = [model '/' char(connection.source)];
    target = [model '/' char(connection.target)];
    sourcePorts = get_param(source, 'PortHandles');
    targetPorts = get_param(target, 'PortHandles');
    add_line(model, sourcePorts.Outport(connection.source_port), ...
        targetPorts.Inport(connection.target_port), 'autorouting', 'on');
end

outputBlock = [model '/' char(spec.output_component)];
outputPorts = get_param(outputBlock, 'PortHandles');
workspacePath = [model '/CircuitPilotOutput'];
add_block('simulink/Sinks/To Workspace', workspacePath, ...
    'VariableName', 'circuitpilot_y', 'SaveFormat', 'Timeseries', ...
    'Position', [500 330 610 360]);
workspacePorts = get_param(workspacePath, 'PortHandles');
add_line(model, outputPorts.Outport(spec.output_port), ...
    workspacePorts.Inport(1), 'autorouting', 'on');

set_param(model, 'StopTime', char(string(spec.stop_time)), ...
    'Solver', 'ode45', 'ReturnWorkspaceOutputs', 'on');
Simulink.BlockDiagram.arrangeSystem(model);
open_system(model);
set_param(model, 'SimulationCommand', 'update');
save_system(model, modelPath);
simOut = sim(model, 'StopTime', char(string(spec.stop_time)));
signal = simOut.get('circuitpilot_y');
time = signal.Time(:);
response = signal.Data(:);
if isempty(time) || numel(time) ~= numel(response) || ...
        any(~isfinite(time)) || any(~isfinite(response))
    error('CircuitPilot:InvalidSimulationOutput', ...
        'Simulation output is empty, malformed, or non-finite.');
end
if ~isfile(modelPath)
    error('CircuitPilot:ModelNotSaved', 'Generated model was not saved: %s', modelPath);
end
result = struct('model_path', modelPath, 'model_opened', true, ...
    'diagram_update', 'succeeded', 'simulation_status', 'succeeded', ...
    'time', time, 'output', response);
resultJson = jsonencode(result);
fid = fopen(resultFile, 'w');
if fid < 0, error('CircuitPilot:ResultWrite', 'Cannot write result file.'); end
fileCleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fwrite(fid, resultJson, 'char');
end

function definitions = circuitpilot_component_definitions()
definitions = struct();
% These are Simulink library paths (not block paths inside the generated model).
definitions.Step = 'simulink/Sources/Step';
definitions.Sum = 'simulink/Math Operations/Sum';
definitions.Gain = 'simulink/Math Operations/Gain';
definitions.TransferFcn = 'simulink/Continuous/Transfer Fcn';
definitions.PIDController = 'simulink/Continuous/PID Controller';
definitions.Integrator = 'simulink/Continuous/Integrator';
definitions.StateSpace = 'simulink/Continuous/State-Space';
definitions.Scope = 'simulink/Sinks/Scope';
end

function close_if_loaded(model)
if bdIsLoaded(model), close_system(model, 0); end
end
