function apf_advanced_optimized()
% Optimized 2D Artificial Potential Field Obstacle Avoidance
% With adaptive parameter tuning and performance optimization
clc;
clear;
close all;
% ==================== SIMULATION PARAMETERS ====================
dt = 0.1; % Time step
max_iterations = 2000; % Maximum number of iterations
goal_threshold = 0.5; % Distance threshold to consider goal reached
% Robot parameters
robot_pos = [0; 0]; % Starting position [x; y]
robot_radius = 0.3; % Robot radius for collision checking
% Goal position
goal_pos = [8; 6]; % Target position
% Obstacles - [x, y, radius]
obstacles = [
3, 2, 0.8;
4, 5, 0.7;
6, 3, 0.6;
2, 4, 0.5;
7, 4.5, 0.7;
];
% ==================== ADVANCED APF PARAMETERS ====================
params = struct();
params.attraction_gain = 0.8; % Ka - Attractive potential gain
params.repulsion_gain = 3.0; % Kr - Repulsive potential gain
params.repulsion_distance = 2.5; % Q* - Distance of influence
params.dt = dt;
% Adaptive field parameters
params.use_adaptive = true;
params.adaptive_gain = 1.5;
params.goal_influence_distance = 3.0;
% Local minima escape parameters
params.enable_escape = true;
params.escape_force_magnitude = 1.2;
params.escape_duration = 30;
params.escape_threshold = 0.3;
% Dynamic parameter adjustment
params.dynamic_tuning = true; % Enable dynamic parameter tuning
params.min_repulsion_gain = 2.0; % Minimum repulsion gain
params.max_repulsion_gain = 5.0; % Maximum repulsion gain
% ==================== INITIALIZATION ====================
% Data logging for analysis
log = initialize_log(robot_pos, goal_pos);
% Local minima detection
stuck_counter = 0;
escape_mode = false;
escape_counter = 0;
escape_direction = [0; 0];
prev_positions = [];
% Performance tracking
performance = struct();
performance.iteration_times = [];
performance.parameter_adjustments = [];
% Create multi-panel figure
figure('Position', [50, 50, 1600, 900]);
% ==================== MAIN SIMULATION LOOP ====================
fprintf('=== Optimized APF Navigation Simulation ===\n');
fprintf('Goal position: (%.1f, %.1f)\n', goal_pos(1), goal_pos(2));
fprintf('Initial position: (%.1f, %.1f)\n', robot_pos(1), robot_pos(2));
fprintf('Number of obstacles: %d\n\n', size(obstacles, 1));
t_start = tic;
for iter = 1:max_iterations
iteration_start = tic;
% Store position for stuck detection
prev_positions = [prev_positions, robot_pos];
if size(prev_positions, 2) > 20
prev_positions = prev_positions(:, 2:end);
end
% Dynamically adjust parameters based on situation
if params.dynamic_tuning
params = dynamic_parameter_tuning(params, robot_pos, goal_pos, obstacles, iter);
end
% Calculate APF forces
[total_force, F_att, F_rep, details] = calculate_advanced_forces(...
robot_pos, goal_pos, obstacles, params, iter);
% Check for local minima
if params.enable_escape && ~escape_mode
[is_stuck, escape_dir] = detect_local_minima_advanced(...
log.velocity, prev_positions, params.escape_threshold);
if is_stuck
fprintf('⚠ Local minima detected at iteration %d! Activating escape.\n', iter);
escape_mode = true;
escape_counter = params.escape_duration;
escape_direction = escape_dir;
stuck_counter = stuck_counter + 1;
end
end
% Apply escape force
if escape_mode
total_force = total_force + params.escape_force_magnitude * escape_direction;
escape_counter = escape_counter - 1;
if escape_counter <= 0
escape_mode = false;
fprintf('✓ Escape completed at iteration %d\n', iter);
end
end
% Update robot position
robot_pos = robot_pos + total_force * dt;
current_time = log.time(end) + dt;
% Log data
log = update_log(log, robot_pos, total_force, current_time, ...
F_att, F_rep, escape_mode, iter, goal_pos);
% Track performance
performance.iteration_times(iter) = toc(iteration_start);
% Check termination conditions
dist_to_goal = log.dist_to_goal(end);
if dist_to_goal < goal_threshold
fprintf('\n✓ GOAL REACHED successfully in %d iterations!\n', iter);
fprintf(' Final distance: %.3f\n', dist_to_goal);
break;
end
if check_collision(robot_pos, robot_radius, obstacles)
fprintf('\n✗ COLLISION DETECTED at iteration %d!\n', iter);
fprintf(' Position: (%.3f, %.3f)\n', robot_pos(1), robot_pos(2));
break;
end
% Update visualization
if mod(iter, 5) == 0 || iter == 1
plot_optimized_visualization(robot_pos, goal_pos, obstacles, log, ...
params, escape_mode, details, robot_radius, iter, performance);
pause(0.01);
end
% Progress report
if mod(iter, 50) == 0
elapsed_time = toc(t_start);
fprintf('Progress: %d iterations, time: %.1fs, dist to goal: %.3f\n', ...
iter, elapsed_time, dist_to_goal);
end
end
total_time = toc(t_start);
% ==================== FINAL ANALYSIS ====================
plot_comprehensive_analysis(log, obstacles, goal_pos, robot_pos, params, ...
robot_radius, performance, total_time);
% Display comprehensive statistics
display_enhanced_statistics(log, obstacles, robot_pos, goal_pos, ...
performance, total_time, stuck_counter);
end
function log = initialize_log(robot_pos, goal_pos)
% Initialize data logging structure
log = struct();
log.position = robot_pos;
log.velocity = [0; 0];
log.time = 0;
log.dist_to_goal = norm(robot_pos - goal_pos);
log.force_att = [0; 0];
log.force_rep = [0; 0];
log.force_total = [0; 0];
log.escape_active = false;
log.iteration = 0;
log.safety_margin = [];
log.parameter_history = [];
end
function log = update_log(log, robot_pos, total_force, current_time, ...
F_att, F_rep, escape_mode, iter, goal_pos)
% Update log with new data
log.position = [log.position, robot_pos];
log.velocity = [log.velocity, total_force];
log.time = [log.time, current_time];
log.dist_to_goal = [log.dist_to_goal, norm(robot_pos - goal_pos)];
log.force_att = [log.force_att, F_att];
log.force_rep = [log.force_rep, F_rep];
log.force_total = [log.force_total, total_force];
log.escape_active = [log.escape_active, escape_mode];
log.iteration = [log.iteration, iter];
end
function params = dynamic_parameter_tuning(params, robot_pos, goal_pos, obstacles, iter)
% Dynamically adjust parameters based on situation
dist_to_goal = norm(robot_pos - goal_pos);
% Find closest obstacle
min_dist = inf;
for i = 1:size(obstacles, 1)
obs_pos = obstacles(i, 1:2)';
obs_radius = obstacles(i, 3);
dist = norm(robot_pos - obs_pos) - obs_radius;
if dist < min_dist
min_dist = dist;
end
end
% Adjust repulsion gain based on proximity to obstacles
if min_dist < 1.0
% Close to obstacles - increase repulsion
params.repulsion_gain = min(params.max_repulsion_gain, ...
params.repulsion_gain * 1.05);
elseif min_dist > 2.0
% Far from obstacles - decrease repulsion
params.repulsion_gain = max(params.min_repulsion_gain, ...
params.repulsion_gain * 0.98);
end
% Adjust attraction gain based on progress
if dist_to_goal < 2.0
% Close to goal - reduce attraction to prevent overshoot
params.attraction_gain = max(0.3, params.attraction_gain * 0.99);
end
end
function [total_force, F_att, F_rep, details] = calculate_advanced_forces(...
robot_pos, goal_pos, obstacles, params, iteration)
% Advanced force calculation with multiple enhancements
% Attractive force with quadratic potential
dist_to_goal = norm(robot_pos - goal_pos);
F_att = params.attraction_gain * (goal_pos - robot_pos);
% Initialize repulsive force
F_rep = [0; 0];
details = struct();
details.closest_obstacle_dist = inf;
details.closest_obstacle_idx = 0;
details.obstacle_forces = [];
details.safety_margin = [];
% Calculate repulsive forces
for i = 1:size(obstacles, 1)
obs_pos = obstacles(i, 1:2)';
obs_radius = obstacles(i, 3);
% Distance calculations
dist_to_obs = norm(robot_pos - obs_pos);
dist_to_surface = max(dist_to_obs - obs_radius, 0.001);
% Track closest obstacle
if dist_to_surface < details.closest_obstacle_dist
details.closest_obstacle_dist = dist_to_surface;
details.closest_obstacle_idx = i;
end
% Calculate repulsive force if within influence zone
if dist_to_surface < params.repulsion_distance
% Direction
direction = (robot_pos - obs_pos) / dist_to_obs;
% Adaptive factor for GNRON prevention
if params.use_adaptive && dist_to_goal < params.goal_influence_distance
adaptive_factor = (dist_to_goal / params.goal_influence_distance)^2;
else
adaptive_factor = 1.0;
end
% Force magnitude with improved formula
magnitude = params.repulsion_gain * adaptive_factor * ...
(1/dist_to_surface - 1/params.repulsion_distance)^2;
% Add force
force = magnitude * direction;
F_rep = F_rep + force;
% Store for analysis
details.obstacle_forces = [details.obstacle_forces; norm(force)];
details.safety_margin = [details.safety_margin; dist_to_surface];
end
end
% Force limiting
max_rep_force = 5.0;
if norm(F_rep) > max_rep_force
F_rep = F_rep / norm(F_rep) * max_rep_force;
end
% Total force
total_force = F_att + F_rep;
% Global force limiting
max_force = 2.5;
if norm(total_force) > max_force
total_force = total_force / norm(total_force) * max_force;
end
% Store details
details.dist_to_goal = dist_to_goal;
details.F_att_mag = norm(F_att);
details.F_rep_mag = norm(F_rep);
end
function [is_stuck, escape_direction] = detect_local_minima_advanced(velocity_history, positions, threshold)
% Advanced local minima detection
is_stuck = false;
escape_direction = [1; 0];
% Need sufficient history
if size(velocity_history, 2) < 15 || size(positions, 2) < 15
return;
end
% Check recent velocities
recent_velocities = velocity_history(:, end-14:end);
avg_velocity = mean(vecnorm(recent_velocities, 2, 1));
% Check position oscillation
if size(positions, 2) >= 10
recent_positions = positions(:, end-9:end);
position_span = max(recent_positions, [], 2) - min(recent_positions, [], 2);
oscillation = norm(position_span);
else
oscillation = inf;
end
% Detect stuck condition
if avg_velocity < threshold && oscillation < 1.0
is_stuck = true;
% Calculate escape direction
if size(positions, 2) >= 5
p_current = positions(:, end);
p_previous = positions(:, max(1, end-4));
movement = p_current - p_previous;
if norm(movement) > 0.01
% Perpendicular direction
escape_direction = [movement(2); -movement(1)] / norm(movement);
% Add randomness
angle = randn * 0.3;
rotation = [cos(angle), -sin(angle); sin(angle), cos(angle)];
escape_direction = rotation * escape_direction;
end
end
end
end
function collision = check_collision(robot_pos, robot_radius, obstacles)
% Check for collisions
collision = false;
for i = 1:size(obstacles, 1)
obs_pos = obstacles(i, 1:2)';
obs_radius = obstacles(i, 3);
if norm(robot_pos - obs_pos) < (robot_radius + obs_radius)
collision = true;
return;
end
end
end
function plot_optimized_visualization(robot_pos, goal_pos, obstacles, log, ...
params, escape_mode, details, robot_radius, iteration, performance)
% Optimized 6-panel visualization
% Panel 1: Robot trajectory and environment
subplot(2, 3, 1);
cla;
hold on;
grid on;
axis equal;
xlim([-1, 10]);
ylim([-1, 8]);
% Plot obstacles
for i = 1:size(obstacles, 1)
obs_x = obstacles(i, 1);
obs_y = obstacles(i, 2);
obs_r = obstacles(i, 3);
% Obstacle body
theta = linspace(0, 2*pi, 50);
x_fill = obs_x + obs_r * cos(theta);
y_fill = obs_y + obs_r * sin(theta);
fill(x_fill, y_fill, [0.5, 0.5, 0.5], 'EdgeColor', 'k', 'LineWidth', 1.5);
% Influence zone
influence_x = obs_x + params.repulsion_distance * cos(theta);
influence_y = obs_y + params.repulsion_distance * sin(theta);
plot(influence_x, influence_y, 'r--', 'LineWidth', 0.8, 'Color', [1, 0, 0, 0.3]);
end
% Highlight closest obstacle
if details.closest_obstacle_idx > 0
idx = details.closest_obstacle_idx;
obs_x = obstacles(idx, 1);
obs_y = obstacles(idx, 2);
obs_r = obstacles(idx, 3);
rectangle('Position', [obs_x-obs_r-0.1, obs_y-obs_r-0.1, ...
2*obs_r+0.2, 2*obs_r+0.2], ...
'Curvature', [1, 1], 'EdgeColor', 'y', 'LineWidth', 2);
end
% Plot goal and path
plot(goal_pos(1), goal_pos(2), 'g*', 'MarkerSize', 20, 'LineWidth', 2);
plot(log.position(1, :), log.position(2, :), 'b-', 'LineWidth', 2);
plot(robot_pos(1), robot_pos(2), 'bo', 'MarkerSize', 12, 'LineWidth', 2.5);
% Plot start point
plot(log.position(1, 1), log.position(2, 1), 'ks', 'MarkerSize', 8, 'LineWidth', 2);
% Force vectors
if size(log.force_total, 2) > 1
F_total = log.force_total(:, end);
if norm(F_total) > 0.01
quiver(robot_pos(1), robot_pos(2), F_total(1), F_total(2), ...
0.8, 'k', 'LineWidth', 3, 'MaxHeadSize', 0.6);
end
end
xlabel('X');
ylabel('Y');
title(sprintf('Trajectory (Iter %d, Dist: %.2f)', iteration, log.dist_to_goal(end)));
legend('Obstacles', 'Influence Zone', 'Goal', 'Path', 'Robot', ...
'Start', 'Total Force', 'Location', 'eastoutside');
% Panel 2: Distance to goal over time
subplot(2, 3, 2);
plot(log.time, log.dist_to_goal, 'b-', 'LineWidth', 2);
grid on;
xlabel('Time (s)');
ylabel('Distance');
title('Distance to Goal');
ylim([0, max(10, max(log.dist_to_goal))]);
% Panel 3: Force magnitudes
subplot(2, 3, 3);
force_att_mag = vecnorm(log.force_att, 2, 1);
force_rep_mag = vecnorm(log.force_rep, 2, 1);
force_total_mag = vecnorm(log.force_total, 2, 1);
plot(log.time, force_att_mag, 'g-', 'LineWidth', 1.5);
hold on;
plot(log.time, force_rep_mag, 'r-', 'LineWidth', 1.5);
plot(log.time, force_total_mag, 'k-', 'LineWidth', 2);
grid on;
xlabel('Time (s)');
ylabel('Force Magnitude');
title('Force Analysis');
legend('F_{att}', 'F_{rep}', 'F_{total}', 'Location', 'best');
% Panel 4: Velocity profile
subplot(2, 3, 4);
velocity_mag = vecnorm(log.velocity, 2, 1);
plot(log.time, velocity_mag, 'm-', 'LineWidth', 2);
hold on;
% Mark escape periods
escape_starts = find(diff([0, log.escape_active]) == 1);
escape_ends = find(diff([log.escape_active, 0]) == -1);
for i = 1:min(length(escape_starts), length(escape_ends))
if escape_starts(i) <= length(log.time) && escape_ends(i) <= length(log.time)
x_fill = [log.time(escape_starts(i)), log.time(escape_ends(i)), ...
log.time(escape_ends(i)), log.time(escape_starts(i))];
y_fill = [0, 0, max(velocity_mag), max(velocity_mag)];
fill(x_fill, y_fill, 'y', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
end
end
grid on;
xlabel('Time (s)');
ylabel('Velocity');
title('Velocity Profile');
% Panel 5: Safety margin
subplot(2, 3, 5);
if isfield(details, 'closest_obstacle_dist')
safety_margin = details.closest_obstacle_dist;
bar(safety_margin, 'b');
hold on;
yline(robot_radius, 'r--', 'Collision Threshold', 'LineWidth', 1.5);
grid on;
xlabel('Current');
ylabel('Distance');
title('Safety Margin');
ylim([0, max(2, safety_margin+0.5)]);
end
% Panel 6: Status panel
subplot(2, 3, 6);
cla;
axis off;
% Calculate stats for display
avg_velocity = mean(velocity_mag(max(1, end-10):end));
status_text = {
'=== SIMULATION STATUS ===';
sprintf('Iteration: %d', iteration);
sprintf('Time: %.1f s', log.time(end));
sprintf('Position: (%.2f, %.2f)', robot_pos(1), robot_pos(2));
sprintf('Distance to goal: %.2f', log.dist_to_goal(end));
sprintf('Velocity: %.2f', velocity_mag(end));
sprintf('Avg velocity: %.2f', avg_velocity);
sprintf('Attractive force: %.2f', norm(log.force_att(:, end)));
sprintf('Repulsive force: %.2f', norm(log.force_rep(:, end)));
sprintf('Safety margin: %.2f', details.closest_obstacle_dist);
'';
'=== PARAMETERS ===';
sprintf('Ka: %.2f', params.attraction_gain);
sprintf('Kr: %.2f', params.repulsion_gain);
sprintf('Q*: %.1f', params.repulsion_distance);
'';
sprintf('Escape mode: %s', string(escape_mode));
sprintf('Iteration time: %.3f s', mean(performance.iteration_times));
};
text(0.1, 0.9, status_text, 'VerticalAlignment', 'top', ...
'FontName', 'Courier', 'FontSize', 10);
drawnow;
end
function plot_comprehensive_analysis(log, obstacles, goal_pos, final_pos, params, ...
robot_radius, performance, total_time)
% Create comprehensive final analysis figure
figure('Position', [100, 100, 1400, 900]);
% Heat map of potential field
subplot(3, 3, [1, 2]);
[X, Y] = meshgrid(-1:0.1:10, -1:0.1:8);
U = zeros(size(X));
for i = 1:size(X, 1)
for j = 1:size(X, 2)
point = [X(i, j); Y(i, j)];
% Attractive potential
U_att = 0.5 * params.attraction_gain * norm(point - goal_pos)^2;
% Repulsive potential
U_rep = 0;
for k = 1:size(obstacles, 1)
obs_pos = obstacles(k, 1:2)';
obs_radius = obstacles(k, 3);
dist = norm(point - obs_pos) - obs_radius;
if dist < params.repulsion_distance && dist > 0
U_rep = U_rep + 0.5 * params.repulsion_gain * ...
(1/dist - 1/params.repulsion_distance)^2;
elseif dist <= 0
U_rep = U_rep + 100;
end
end
U(i, j) = U_att + U_rep;
end
end
imagesc(X(1,:), Y(:,1), log10(U+1));
colormap('jet');
colorbar;
hold on;
plot(log.position(1, :), log.position(2, :), 'w-', 'LineWidth', 2.5);
plot(goal_pos(1), goal_pos(2), 'w*', 'MarkerSize', 20, 'LineWidth', 2.5);
plot(log.position(1, 1), log.position(2, 1), 'ws', 'MarkerSize', 15, 'LineWidth', 2.5);
title('Potential Field Heat Map with Trajectory (log scale)');
xlabel('X');
ylabel('Y');
axis equal tight;
% Distance to goal over time
subplot(3, 3, 3);
plot(log.time, log.dist_to_goal, 'b-', 'LineWidth', 2.5);
grid on;
xlabel('Time (s)');
ylabel('Distance');
title('Convergence');
% Force evolution
subplot(3, 3, 4);
force_att_mag = vecnorm(log.force_att, 2, 1);
force_rep_mag = vecnorm(log.force_rep, 2, 1);
plot(log.time, force_att_mag, 'g-', 'LineWidth', 2);
hold on;
plot(log.time, force_rep_mag, 'r-', 'LineWidth', 2);
grid on;
xlabel('Time (s)');
ylabel('Force');
title('Force Evolution');
legend('Attractive', 'Repulsive', 'Location', 'best');
% Velocity profile
subplot(3, 3, 5);
velocity_mag = vecnorm(log.velocity, 2, 1);
plot(log.time, velocity_mag, 'm-', 'LineWidth', 2);
grid on;
xlabel('Time (s)');
ylabel('Velocity');
title('Velocity Profile');
% Safety margin
subplot(3, 3, 6);
safety_distances = calculate_safety_distances(log.position, obstacles);
plot(log.time, safety_distances, 'b-', 'LineWidth', 2);
hold on;
yline(robot_radius, 'r--', 'Collision Threshold', 'LineWidth', 1.5);
grid on;
xlabel('Time (s)');
ylabel('Distance');
title('Safety Margin');
legend('Safety Margin', 'Threshold', 'Location', 'best');
% Trajectory components
subplot(3, 3, 7);
plot(log.time, log.position(1, :), 'b-', 'LineWidth', 2);
hold on;
plot(log.time, log.position(2, :), 'r-', 'LineWidth', 2);
grid on;
xlabel('Time (s)');
ylabel('Position');
title('Trajectory Components');
legend('X', 'Y', 'Location', 'best');
% Performance metrics
subplot(3, 3, 8);
if length(performance.iteration_times) > 1
plot(1:length(performance.iteration_times), performance.iteration_times * 1000, 'k-', 'LineWidth', 1.5);
grid on;
xlabel('Iteration');
ylabel('Time (ms)');
title('Computation Time per Iteration');
end
% Summary statistics
subplot(3, 3, 9);
axis off;
path_length = calculate_path_length(log.position);
straight_line = norm(goal_pos - log.position(:, 1));
efficiency = 100 * straight_line / path_length;
summary_text = {
'=== SUMMARY STATISTICS ===';
sprintf('Total time: %.2f s', total_time);
sprintf('Iterations: %d', length(log.iteration));
sprintf('Path length: %.2f', path_length);
sprintf('Efficiency: %.1f%%', efficiency);
sprintf('Min safety: %.2f', min(safety_distances));
sprintf('Avg safety: %.2f', mean(safety_distances));
sprintf('Avg velocity: %.2f', mean(velocity_mag));
sprintf('Max velocity: %.2f', max(velocity_mag));
sprintf('Escape count: %d', sum(diff([0, log.escape_active]) == 1));
};
text(0.1, 0.9, summary_text, 'VerticalAlignment', 'top', ...
'FontName', 'Courier', 'FontSize', 11, 'FontWeight', 'bold');
end
function safety_distances = calculate_safety_distances(positions, obstacles)
% Calculate minimum distances to obstacles throughout trajectory
num_points = size(positions, 2);
safety_distances = zeros(1, num_points);
for i = 1:num_points
pos = positions(:, i);
min_dist = inf;
for j = 1:size(obstacles, 1)
obs_pos = obstacles(j, 1:2)';
obs_radius = obstacles(j, 3);
dist = norm(pos - obs_pos) - obs_radius;
min_dist = min(min_dist, dist);
end
safety_distances(i) = min_dist;
end
end
function display_enhanced_statistics(log, obstacles, final_pos, goal_pos, ...
performance, total_time, escape_count)
% Display enhanced comprehensive simulation statistics
fprintf('\n%s\n', repmat('=', 70));
fprintf('OPTIMIZED APF SIMULATION - FINAL STATISTICS\n');
fprintf('%s\n', repmat('=', 70));
% Basic metrics
fprintf('\n📊 BASIC METRICS:\n');
fprintf(' • Total iterations: %d\n', length(log.iteration));
fprintf(' • Total time: %.2f seconds\n', total_time);
fprintf(' • Computation time per iteration: %.2f ms\n', ...
1000 * mean(performance.iteration_times));
fprintf(' • Path length: %.3f units\n', calculate_path_length(log.position));
fprintf(' • Final position: (%.3f, %.3f)\n', final_pos(1), final_pos(2));
fprintf(' • Final distance to goal: %.3f\n', log.dist_to_goal(end));
% Performance metrics
fprintf('\n⚡ PERFORMANCE METRICS:\n');
velocity_mag = vecnorm(log.velocity, 2, 1);
fprintf(' • Average velocity: %.3f units/s\n', mean(velocity_mag));
fprintf(' • Max velocity: %.3f units/s\n', max(velocity_mag));
fprintf(' • Average attractive force: %.3f\n', mean(vecnorm(log.force_att, 2, 1)));
fprintf(' • Average repulsive force: %.3f\n', mean(vecnorm(log.force_rep, 2, 1)));
% Efficiency metrics
fprintf('\n🎯 EFFICIENCY METRICS:\n');
straight_line_dist = norm(goal_pos - log.position(:, 1));
path_length = calculate_path_length(log.position);
fprintf(' • Straight-line distance: %.3f units\n', straight_line_dist);
fprintf(' • Path efficiency: %.1f%%\n', 100 * straight_line_dist / path_length);
fprintf(' • Convergence rate: %.3f units/iteration\n', ...
(log.dist_to_goal(1) - log.dist_to_goal(end)) / length(log.iteration));
% Safety metrics
fprintf('\n🛡️ SAFETY METRICS:\n');
safety_distances = calculate_safety_distances(log.position, obstacles);
fprintf(' • Minimum distance to obstacles: %.3f units\n', min(safety_distances));
fprintf(' • Average safety margin: %.3f units\n', mean(safety_distances));
fprintf(' • Safety margin variance: %.3f\n', var(safety_distances));
fprintf(' • Number of escape activations: %d\n', escape_count);
% Additional metrics
fprintf('\n📈 ADDITIONAL METRICS:\n');
fprintf(' • Path smoothness: %.3f\n', calculate_path_smoothness(log.position));
fprintf(' • Force balance ratio: %.2f\n', ...
mean(vecnorm(log.force_rep, 2, 1)) / mean(vecnorm(log.force_att, 2, 1)));
fprintf('%s\n', repmat('=', 70));
end
function smoothness = calculate_path_smoothness(positions)
% Calculate path smoothness (lower is smoother)
if size(positions, 2) < 3
smoothness = 0;
return;
end
angles = [];
for i = 2:size(positions, 2)-1
v1 = positions(:, i) - positions(:, i-1);
v2 = positions(:, i+1) - positions(:, i);
if norm(v1) > 0 && norm(v2) > 0
cos_angle = dot(v1, v2) / (norm(v1) * norm(v2));
cos_angle = max(-1, min(1, cos_angle)); % Clamp to valid range
angles = [angles, acos(cos_angle)];
end
end
smoothness = mean(angles) * 180 / pi; % Average turning angle in degrees
end
function path_length = calculate_path_length(positions)
% Calculate total path length
path_length = 0;
for i = 2:size(positions, 2)
path_length = path_length + norm(positions(:, i) - positions(:, i-1));
end
end

