function apf_3d_enhanced_navigation()
    %  3D Artificial Potential Field Obstacle Avoidance
    %  proper obstacle placement and multi-path navigation
    
    clc;
    clear;
    close all;
    
    % ==================== SIMULATION PARAMETERS ====================
    dt = 0.1;                    % Time step
    max_iterations = 3000;        % Maximum number of iterations
    goal_threshold = 0.8;         % Distance threshold to consider goal reached
    
    % Robot parameters
    robot_pos = [1; 1; 2];        % Starting position [x; y; z]
    robot_radius = 0.4;           % Robot radius for collision checking
    
    % Goal position
    goal_pos = [9; 8; 3];         % Target position
    
    % ==================== OBSTACLES ====================
    % Each obstacle: [x, y, z, width, depth, height, type]
    % type: 1 = cube, 2 = rectangular prism
    obstacles = [
        % Left side obstacles (create left corridor)
        %2.5, 2.5, 2.0, 1.8, 1.8, 4.0, 1;    % Cube left-front
        %2.5, 6.5, 2.0, 1.8, 1.8, 4.5, 1;    % Cube left-back
        
        % Right side obstacles (create right corridor)
        7.5, 2.5, 2.0, 1.8, 1.8, 4.0, 1;    % Cube right-front
        %7.5, 6.5, 2.0, 1.8, 1.8, 4.5, 1;    % Cube right-back
        
        % Center obstacles - create multiple paths
        %4.5, 4.5, 1.5, 2.0, 2.0, 2.5, 1;    % Center cube (low) - can go over
        6.0, 5.0, 4.0, 2.5, 1.5, 2.0, 2;    % Horizontal rectangle (high) - go under
        3.0, 7.0, 3.5, 1.5, 2.5, 2.5, 2;    % Vertical rectangle (mid) - go around
        
        % Additional obstacles to create maze-like environment
        5.0, 2.0, 2.5, 2.0, 1.0, 4.0, 2;    % Wall obstacle front
        5.0, 8.0, 2.5, 2.0, 1.0, 4.0, 2;    % Wall obstacle back
    ];
    
    % ==================== APF PARAMETERS ====================
    params = struct();
    params.attraction_gain = 0.7;       % Ka - Attractive potential gain
    params.repulsion_gain = 3.5;         % Kr - Repulsive potential gain
    params.repulsion_distance = 3.0;     % Q* - Distance of influence
    params.dt = dt;
    
    % Adaptive field parameters
    params.use_adaptive = true;
    params.adaptive_gain = 1.5;
    params.goal_influence_distance = 4.0;
    
    % Local minima escape parameters
    params.enable_escape = true;
    params.escape_force_magnitude = 1.8;
    params.escape_duration = 40;
    params.escape_threshold = 0.25;
    
    % Dynamic parameter adjustment
    params.dynamic_tuning = true;
    params.min_repulsion_gain = 2.5;
    params.max_repulsion_gain = 5.5;
    
    % ==================== INITIALIZATION ====================
    log = initialize_log_3d(robot_pos, goal_pos);
    
    % Local minima detection
    stuck_counter = 0;
    escape_mode = false;
    escape_counter = 0;
    escape_direction = [0; 0; 0];
    prev_positions = [];
    
    % Performance tracking
    performance = struct();
    performance.iteration_times = [];
    
    % Create figure
    figure('Position', [50, 50, 1800, 900]);
    
    % ==================== MAIN SIMULATION LOOP ====================
    fprintf('=== Enhanced 3D APF Navigation Simulation ===\n');
    fprintf('Goal position: (%.1f, %.1f, %.1f)\n', goal_pos(1), goal_pos(2), goal_pos(3));
    fprintf('Initial position: (%.1f, %.1f, %.1f)\n', robot_pos(1), robot_pos(2), robot_pos(3));
    fprintf('Number of obstacles: %d\n\n', size(obstacles, 1));
    
    t_start = tic;
    
    for iter = 1:max_iterations
        iteration_start = tic;
        
        % Store position for stuck detection
        prev_positions = [prev_positions, robot_pos];
        if size(prev_positions, 2) > 20
            prev_positions = prev_positions(:, 2:end);
        end
        
        % Dynamically adjust parameters
        if params.dynamic_tuning
            params = dynamic_parameter_tuning_3d(params, robot_pos, goal_pos, obstacles, iter);
        end
        
        % Calculate APF forces with corridor detection
        [total_force, F_att, F_rep, details] = calculate_enhanced_forces_3d(...
            robot_pos, goal_pos, obstacles, params, iter);
        
        % Check for local minima
        if params.enable_escape && ~escape_mode
            [is_stuck, escape_dir] = detect_local_minima_3d(...
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
        
        % Add small random perturbation to help find corridors
        if mod(iter, 20) == 0 && ~escape_mode && norm(total_force) < 0.5
            random_force = 0.2 * randn(3, 1);
            total_force = total_force + random_force;
        end
        
        % Update robot position
        robot_pos = robot_pos + total_force * dt;
        current_time = log.time(end) + dt;
        
        % Log data
        log = update_log_3d(log, robot_pos, total_force, current_time, ...
                           F_att, F_rep, escape_mode, iter, goal_pos, details);
        
        % Track performance
        performance.iteration_times(iter) = toc(iteration_start);
        
        % Check termination conditions
        dist_to_goal = log.dist_to_goal(end);
        if dist_to_goal < goal_threshold
            fprintf('\n✓ GOAL REACHED successfully in %d iterations!\n', iter);
            fprintf('  Final distance: %.3f\n', dist_to_goal);
            break;
        end
        
        if check_collision_3d(robot_pos, robot_radius, obstacles)
            fprintf('\n✗ COLLISION DETECTED at iteration %d!\n', iter);
            fprintf('  Position: (%.3f, %.3f, %.3f)\n', robot_pos(1), robot_pos(2), robot_pos(3));
            break;
        end
        
        % Update visualization
        if mod(iter, 5) == 0 || iter == 1
            plot_enhanced_3d_visualization(robot_pos, goal_pos, obstacles, log, ...
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
    plot_enhanced_3d_final_analysis(log, obstacles, goal_pos, robot_pos, params, ...
                                   robot_radius, performance, total_time, stuck_counter);
    
    % Display statistics
    display_enhanced_3d_statistics(log, obstacles, robot_pos, goal_pos, ...
                                  performance, total_time, stuck_counter);
end

function log = initialize_log_3d(robot_pos, goal_pos)
    % Initialize 3D data logging
    log = struct();
    log.position = robot_pos;
    log.velocity = [0; 0; 0];
    log.time = 0;
    log.dist_to_goal = norm(robot_pos - goal_pos);
    log.force_att = [0; 0; 0];
    log.force_rep = [0; 0; 0];
    log.force_total = [0; 0; 0];
    log.escape_active = false;
    log.iteration = 0;
    log.closest_obstacle_dist = [];
    log.closest_obstacle_type = [];
    log.corridor_score = [];
end

function log = update_log_3d(log, robot_pos, total_force, current_time, ...
                            F_att, F_rep, escape_mode, iter, goal_pos, details)
    % Update 3D log with new data
    log.position = [log.position, robot_pos];
    log.velocity = [log.velocity, total_force];
    log.time = [log.time, current_time];
    log.dist_to_goal = [log.dist_to_goal, norm(robot_pos - goal_pos)];
    log.force_att = [log.force_att, F_att];
    log.force_rep = [log.force_rep, F_rep];
    log.force_total = [log.force_total, total_force];
    log.escape_active = [log.escape_active, escape_mode];
    log.iteration = [log.iteration, iter];
    log.closest_obstacle_dist = [log.closest_obstacle_dist, details.closest_obstacle_dist];
    log.closest_obstacle_type = [log.closest_obstacle_type, details.closest_obstacle_type];
    log.corridor_score = [log.corridor_score, details.corridor_score];
end

function params = dynamic_parameter_tuning_3d(params, robot_pos, goal_pos, obstacles, iter)
    % Dynamically adjust 3D parameters
    
    dist_to_goal = norm(robot_pos - goal_pos);
    
    % Find closest obstacle distance
    min_dist = inf;
    for i = 1:size(obstacles, 1)
        dist = distance_to_obstacle_3d(robot_pos, obstacles(i, :));
        if dist < min_dist
            min_dist = dist;
        end
    end
    
    % Adjust repulsion gain based on proximity
    if min_dist < 1.2
        params.repulsion_gain = min(params.max_repulsion_gain, ...
                                   params.repulsion_gain * 1.02);
    elseif min_dist > 2.5
        params.repulsion_gain = max(params.min_repulsion_gain, ...
                                   params.repulsion_gain * 0.99);
    end
    
    % Adjust attraction gain near goal
    if dist_to_goal < 3.0
        params.attraction_gain = max(0.4, params.attraction_gain * 0.98);
    end
end

function dist = distance_to_obstacle_3d(point, obstacle)
    % Calculate minimum distance from point to rectangular obstacle
    
    obs_x = obstacle(1);
    obs_y = obstacle(2);
    obs_z = obstacle(3);
    obs_w = obstacle(4);
    obs_d = obstacle(5);
    obs_h = obstacle(6);
    
    % Calculate distance to obstacle boundaries
    dx = max(abs(point(1) - obs_x) - obs_w/2, 0);
    dy = max(abs(point(2) - obs_y) - obs_d/2, 0);
    dz = max(abs(point(3) - obs_z) - obs_h/2, 0);
    
    % Euclidean distance to obstacle surface
    dist = sqrt(dx^2 + dy^2 + dz^2);
end

function [total_force, F_att, F_rep, details] = calculate_enhanced_forces_3d(...
    robot_pos, goal_pos, obstacles, params, iteration)
    % Enhanced 3D force calculation with corridor detection
    
    % Attractive force
    dist_to_goal = norm(robot_pos - goal_pos);
    F_att = params.attraction_gain * (goal_pos - robot_pos);
    
    % Initialize repulsive force
    F_rep = [0; 0; 0];
    details = struct();
    details.closest_obstacle_dist = inf;
    details.closest_obstacle_idx = 0;
    details.closest_obstacle_type = 0;
    details.obstacle_forces = [];
    details.corridor_score = 0;
    
    % Calculate repulsive forces from each obstacle
    for i = 1:size(obstacles, 1)
        obstacle = obstacles(i, :);
        
        % Calculate distance and direction
        [dist_to_surface, direction] = distance_and_direction_to_obstacle_3d(robot_pos, obstacle);
        
        % Track closest obstacle
        if dist_to_surface < details.closest_obstacle_dist
            details.closest_obstacle_dist = dist_to_surface;
            details.closest_obstacle_idx = i;
            details.closest_obstacle_type = obstacle(7);
        end
        
        % Calculate repulsive force if within influence zone
        if dist_to_surface < params.repulsion_distance && dist_to_surface > 0.05
            % Adaptive factor
            if params.use_adaptive && dist_to_goal < params.goal_influence_distance
                adaptive_factor = (dist_to_goal / params.goal_influence_distance)^2;
            else
                adaptive_factor = 1.0;
            end
            
            % Force magnitude
            magnitude = params.repulsion_gain * adaptive_factor * ...
                       (1/dist_to_surface - 1/params.repulsion_distance)^2;
            
            % Add force
            F_rep = F_rep + magnitude * direction;
            
            % Store for analysis
            details.obstacle_forces = [details.obstacle_forces; norm(magnitude * direction)];
        end
    end
    
    % Add height preference for multi-path navigation
    if robot_pos(3) < 2.0 && details.closest_obstacle_dist > 1.0
        % Lower height - encourage staying low for under paths
        F_rep(3) = F_rep(3) - 0.1;
    elseif robot_pos(3) > 5.0
        % Too high - encourage going down if possible
        F_rep(3) = F_rep(3) + 0.2;
    end
    
    % Calculate corridor score
    details.corridor_score = calculate_corridor_score(robot_pos, obstacles, params.repulsion_distance);
    
    % Force limiting
    max_rep_force = 5.0;
    if norm(F_rep) > max_rep_force
        F_rep = F_rep / norm(F_rep) * max_rep_force;
    end
    
    % Total force
    total_force = F_att + F_rep;
    
    % Global force limiting
    max_force = 3.0;
    if norm(total_force) > max_force
        total_force = total_force / norm(total_force) * max_force;
    end
    
    % Store details
    details.dist_to_goal = dist_to_goal;
    details.F_att_mag = norm(F_att);
    details.F_rep_mag = norm(F_rep);
end

function [dist, direction] = distance_and_direction_to_obstacle_3d(point, obstacle)
    % Calculate distance and direction to rectangular obstacle
    
    obs_x = obstacle(1);
    obs_y = obstacle(2);
    obs_z = obstacle(3);
    obs_w = obstacle(4);
    obs_d = obstacle(5);
    obs_h = obstacle(6);
    
    % Find closest point on obstacle surface
    closest = point;
    
    % Clamp to obstacle boundaries
    closest(1) = max(obs_x - obs_w/2, min(obs_x + obs_w/2, point(1)));
    closest(2) = max(obs_y - obs_d/2, min(obs_y + obs_d/2, point(2)));
    closest(3) = max(obs_z - obs_h/2, min(obs_z + obs_h/2, point(3)));
    
    % If point is inside obstacle, push to nearest surface
    if point(1) > obs_x - obs_w/2 && point(1) < obs_x + obs_w/2 && ...
       point(2) > obs_y - obs_d/2 && point(2) < obs_y + obs_d/2 && ...
       point(3) > obs_z - obs_h/2 && point(3) < obs_z + obs_h/2
        
        % Point inside obstacle - find closest surface
        dist_to_surfaces = [
            abs(point(1) - (obs_x - obs_w/2));
            abs(point(1) - (obs_x + obs_w/2));
            abs(point(2) - (obs_y - obs_d/2));
            abs(point(2) - (obs_y + obs_d/2));
            abs(point(3) - (obs_z - obs_h/2));
            abs(point(3) - (obs_z + obs_h/2))
        ];
        
        [~, min_idx] = min(dist_to_surfaces);
        
        switch min_idx
            case 1, closest(1) = obs_x - obs_w/2;
            case 2, closest(1) = obs_x + obs_w/2;
            case 3, closest(2) = obs_y - obs_d/2;
            case 4, closest(2) = obs_y + obs_d/2;
            case 5, closest(3) = obs_z - obs_h/2;
            case 6, closest(3) = obs_z + obs_h/2;
        end
    end
    
    % Calculate distance and direction
    diff = point - closest;
    dist = norm(diff);
    
    if dist > 0.001
        direction = diff / dist;
    else
        direction = [1; 0; 0];
    end
end

function score = calculate_corridor_score(robot_pos, obstacles, influence_dist)
    % Calculate how much free space is around the robot
    
    % Sample directions (8 main directions)
    directions = [
        1, 0, 0; -1, 0, 0; 0, 1, 0; 0, -1, 0; 0, 0, 1; 0, 0, -1;
        0.707, 0.707, 0; -0.707, 0.707, 0; 0.707, -0.707, 0; -0.707, -0.707, 0
    ]';
    
    min_free_dist = influence_dist;
    
    for d = 1:size(directions, 2)
        dir = directions(:, d);
        
        % Check distance in this direction
        for step = 0.2:0.2:influence_dist
            check_point = robot_pos + step * dir;
            
            % Check if this point is near any obstacle
            for i = 1:size(obstacles, 1)
                dist = distance_to_obstacle_3d(check_point, obstacles(i, :));
                if dist < 0.3  % Close to obstacle
                    if step < min_free_dist
                        min_free_dist = step;
                    end
                    break;
                end
            end
        end
    end
    
    % Normalize score (0 = trapped, 1 = wide open)
    score = min(1, min_free_dist / (influence_dist * 0.6));
end

function [is_stuck, escape_direction] = detect_local_minima_3d(velocity_history, positions, threshold)
    % Detect if robot is stuck in 3D local minimum
    
    is_stuck = false;
    escape_direction = [1; 0; 0];
    
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
    if avg_velocity < threshold && oscillation < 1.2
        is_stuck = true;
        
        % Generate random escape direction in horizontal plane 
        angle = rand * 2 * pi;
        escape_direction = [cos(angle); sin(angle); (rand - 0.5) * 0.5];
        escape_direction = escape_direction / norm(escape_direction);
    end
end

function collision = check_collision_3d(robot_pos, robot_radius, obstacles)
    % Check for collisions with rectangular obstacles
    collision = false;
    
    for i = 1:size(obstacles, 1)
        dist = distance_to_obstacle_3d(robot_pos, obstacles(i, :));
        
        if dist < robot_radius
            collision = true;
            return;
        end
    end
end

function plot_enhanced_3d_visualization(robot_pos, goal_pos, obstacles, log, ...
    params, escape_mode, details, robot_radius, iteration, performance)
    % Enhanced 3D visualization
    
    % Main 3D view
    subplot(2, 4, [1, 2, 5, 6]);
    cla;
    hold on;
    grid on;
    view(45, 25);
    
    % Set axis limits
    xlim([0, 11]);
    ylim([0, 10]);
    zlim([0, 8]);
    
    % Plot obstacles
    for i = 1:size(obstacles, 1)
        obs = obstacles(i, :);
        highlight = (i == details.closest_obstacle_idx);
        plot_rectangular_obstacle(obs, highlight);
    end
    
    % Plot goal
    plot3(goal_pos(1), goal_pos(2), goal_pos(3), 'g*', ...
          'MarkerSize', 20, 'LineWidth', 2);
    
    % Plot path
    if size(log.position, 2) > 1
        plot3(log.position(1, :), log.position(2, :), log.position(3, :), ...
              'b-', 'LineWidth', 2);
    end
    
    % Plot current robot position
    plot3(robot_pos(1), robot_pos(2), robot_pos(3), 'bo', ...
          'MarkerSize', 15, 'LineWidth', 2.5, 'MarkerFaceColor', 'b');
    
    % Plot start point
    plot3(log.position(1, 1), log.position(2, 1), log.position(3, 1), ...
          'ks', 'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor', 'k');
    
    % Force vector
    if size(log.force_total, 2) > 1
        F_total = log.force_total(:, end);
        if norm(F_total) > 0.01
            quiver3(robot_pos(1), robot_pos(2), robot_pos(3), ...
                    F_total(1), F_total(2), F_total(3), ...
                    0.8, 'k', 'LineWidth', 2.5, 'MaxHeadSize', 0.5);
        end
    end
    
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    title(sprintf('3D Navigation (Iter %d, Dist: %.2f)', ...
          iteration, log.dist_to_goal(end)));
    
    legend('Obstacles', 'Goal', 'Path', 'Robot', 'Start', 'Force', ...
           'Location', 'eastoutside');
    
    % Distance to goal plot
    subplot(2, 4, 3);
    if length(log.time) == length(log.dist_to_goal)
        plot(log.time, log.dist_to_goal, 'b-', 'LineWidth', 2);
        grid on;
        xlabel('Time (s)');
        ylabel('Distance');
        title('Distance to Goal');
    end
    
    % Force magnitudes
    subplot(2, 4, 4);
    if size(log.force_att, 2) > 1 && size(log.force_rep, 2) > 1
        force_att_mag = vecnorm(log.force_att, 2, 1);
        force_rep_mag = vecnorm(log.force_rep, 2, 1);
        
        if length(log.time) == length(force_att_mag)
            plot(log.time, force_att_mag, 'g-', 'LineWidth', 1.5);
            hold on;
            plot(log.time, force_rep_mag, 'r-', 'LineWidth', 1.5);
            grid on;
            xlabel('Time (s)');
            ylabel('Force');
            title('Force Analysis');
            legend('F_{att}', 'F_{rep}', 'Location', 'best');
        end
    end
    
    % Corridor score
    subplot(2, 4, 7);
    if ~isempty(log.corridor_score) && length(log.corridor_score) > 1
        if length(log.time) == length(log.corridor_score)
            plot(log.time, log.corridor_score, 'm-', 'LineWidth', 2);
            grid on;
            xlabel('Time (s)');
            ylabel('Score');
            title('Corridor Availability');
            ylim([0, 1]);
        end
    end
    
    % Height profile
    subplot(2, 4, 8);
    if size(log.position, 2) > 1
        if length(log.time) == size(log.position, 2)
            plot(log.time, log.position(3, :), 'c-', 'LineWidth', 2);
            grid on;
            xlabel('Time (s)');
            ylabel('Height');
            title('Height Profile');
        end
    end
    
    drawnow;
end

function plot_rectangular_obstacle(obstacle, highlight)
    % Plot a rectangular obstacle
    
    x = obstacle(1);
    y = obstacle(2);
    z = obstacle(3);
    w = obstacle(4);
    d = obstacle(5);
    h = obstacle(6);
    type = obstacle(7);
    
    % Define vertices
    vertices = [
        x - w/2, y - d/2, z - h/2;
        x + w/2, y - d/2, z - h/2;
        x + w/2, y + d/2, z - h/2;
        x - w/2, y + d/2, z - h/2;
        x - w/2, y - d/2, z + h/2;
        x + w/2, y - d/2, z + h/2;
        x + w/2, y + d/2, z + h/2;
        x - w/2, y + d/2, z + h/2
    ];
    
    % Define faces
    faces = [
        1, 2, 6, 5;  % bottom
        2, 3, 7, 6;  % front
        3, 4, 8, 7;  % right
        4, 1, 5, 8;  % back
        1, 2, 3, 4;  % top
        5, 6, 7, 8   % left
    ];
    
    % Choose color based on type
    if type == 1
        face_color = [0.7, 0.7, 0.7];  % Gray for cubes
    else
        face_color = [0.5, 0.7, 0.9];  % Light blue for rectangles
    end
    
    % Highlight if closest
    if highlight
        edge_color = 'y';
        line_width = 2.5;
        face_alpha = 0.7;
    else
        edge_color = 'k';
        line_width = 1;
        face_alpha = 0.5;
    end
    
    % Plot the obstacle
    patch('Vertices', vertices, 'Faces', faces, ...
          'FaceColor', face_color, 'EdgeColor', edge_color, ...
          'LineWidth', line_width, 'FaceAlpha', face_alpha);
end

function plot_enhanced_3d_final_analysis(log, obstacles, goal_pos, final_pos, params, ...
                                        robot_radius, performance, total_time, escape_count)
    % Final comprehensive 3D analysis
    
    figure('Position', [100, 100, 1600, 1000]);
    
    % 3D trajectory with obstacles
    subplot(2, 3, [1, 2, 4, 5]);
    hold on;
    grid on;
    view(45, 30);
    
    % Plot all obstacles
    for i = 1:size(obstacles, 1)
        plot_rectangular_obstacle(obstacles(i, :), false);
    end
    
    % Plot trajectory
    if size(log.position, 2) > 1
        plot3(log.position(1, :), log.position(2, :), log.position(3, :), ...
              'b-', 'LineWidth', 2.5);
    end
    
    % Plot start, final, and goal
    plot3(log.position(1, 1), log.position(2, 1), log.position(3, 1), ...
          'ks', 'MarkerSize', 15, 'LineWidth', 2.5, 'MarkerFaceColor', 'k');
    plot3(final_pos(1), final_pos(2), final_pos(3), ...
          'bo', 'MarkerSize', 15, 'LineWidth', 2.5, 'MarkerFaceColor', 'b');
    plot3(goal_pos(1), goal_pos(2), goal_pos(3), ...
          'g*', 'MarkerSize', 25, 'LineWidth', 2.5);
    
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    title('3D Navigation Path');
    legend('Obstacles', 'Path', 'Start', 'Final', 'Goal', 'Location', 'best');
    axis equal;
    xlim([0, 11]);
    ylim([0, 10]);
    zlim([0, 8]);
    
    % Distance to goal over time
    subplot(2, 3, 3);
    if length(log.time) == length(log.dist_to_goal)
        plot(log.time, log.dist_to_goal, 'b-', 'LineWidth', 2);
        grid on;
        xlabel('Time (s)');
        ylabel('Distance');
        title('Convergence');
    end
    
    % Force evolution
    subplot(2, 3, 6);
    if size(log.force_att, 2) > 1 && size(log.force_rep, 2) > 1
        force_att_mag = vecnorm(log.force_att, 2, 1);
        force_rep_mag = vecnorm(log.force_rep, 2, 1);
        
        if length(log.time) == length(force_att_mag)
            plot(log.time, force_att_mag, 'g-', 'LineWidth', 2);
            hold on;
            plot(log.time, force_rep_mag, 'r-', 'LineWidth', 2);
            grid on;
            xlabel('Time (s)');
            ylabel('Force');
            title('Force Evolution');
            legend('Attractive', 'Repulsive', 'Location', 'best');
        end
    end
    
    % Add summary text
    path_length = calculate_path_length_3d(log.position);
    straight_line = norm(goal_pos - log.position(:, 1));
    efficiency = 100 * straight_line / path_length;
    
    annotation('textbox', [0.15, 0.02, 0.7, 0.05], 'String', ...
        sprintf('Total Time: %.2fs | Iterations: %d | Path Length: %.2f | Efficiency: %.1f%% | Escapes: %d', ...
        total_time, length(log.iteration), path_length, efficiency, escape_count), ...
        'HorizontalAlignment', 'center', 'FontSize', 12, ...
        'BackgroundColor', 'w', 'EdgeColor', 'k');
end

function display_enhanced_3d_statistics(log, obstacles, final_pos, goal_pos, ...
                                       performance, total_time, escape_count)
    % Display 3D simulation statistics
    
    fprintf('\n%s\n', repmat('=', 70));
    fprintf('3D APF SIMULATION - FINAL STATISTICS\n');
    fprintf('%s\n', repmat('=', 70));
    
    % Basic metrics
    fprintf('\n📊 BASIC METRICS:\n');
    fprintf('  • Total iterations: %d\n', length(log.iteration));
    fprintf('  • Total time: %.2f seconds\n', total_time);
    fprintf('  • Computation time per iteration: %.2f ms\n', ...
            1000 * mean(performance.iteration_times(performance.iteration_times > 0)));
    fprintf('  • Path length: %.3f units\n', calculate_path_length_3d(log.position));
    fprintf('  • Final position: (%.3f, %.3f, %.3f)\n', ...
            final_pos(1), final_pos(2), final_pos(3));
    fprintf('  • Final distance to goal: %.3f\n', log.dist_to_goal(end));
    
    % Performance metrics
    fprintf('\n⚡ PERFORMANCE METRICS:\n');
    velocity_mag = vecnorm(log.velocity, 2, 1);
    fprintf('  • Average velocity: %.3f units/s\n', mean(velocity_mag));
    fprintf('  • Max velocity: %.3f units/s\n', max(velocity_mag));
    fprintf('  • Average attractive force: %.3f\n', mean(vecnorm(log.force_att, 2, 1)));
    fprintf('  • Average repulsive force: %.3f\n', mean(vecnorm(log.force_rep, 2, 1)));
    
    % Efficiency metrics
    fprintf('\n🎯 EFFICIENCY METRICS:\n');
    straight_line_dist = norm(goal_pos - log.position(:, 1));
    path_length = calculate_path_length_3d(log.position);
    fprintf('  • Straight-line distance: %.3f units\n', straight_line_dist);
    fprintf('  • Path efficiency: %.1f%%\n', 100 * straight_line_dist / path_length);
    fprintf('  • Convergence rate: %.3f units/iteration\n', ...
            (log.dist_to_goal(1) - log.dist_to_goal(end)) / length(log.iteration));
    
    % Safety metrics
    fprintf('\n🛡️ SAFETY METRICS:\n');
    if ~isempty(log.closest_obstacle_dist)
        fprintf('  • Minimum distance to obstacles: %.3f units\n', min(log.closest_obstacle_dist));
        fprintf('  • Average safety margin: %.3f units\n', mean(log.closest_obstacle_dist));
        fprintf('  • Number of escape activations: %d\n', escape_count);
    end
    
    % Corridor metrics
    fprintf('\n🛣️  CORRIDOR METRICS:\n');
    if ~isempty(log.corridor_score)
        fprintf('  • Average corridor score: %.3f\n', mean(log.corridor_score));
        fprintf('  • Max corridor score: %.3f\n', max(log.corridor_score));
        fprintf('  • Min corridor score: %.3f\n', min(log.corridor_score));
    end
    
    fprintf('%s\n', repmat('=', 70));
end

function path_length = calculate_path_length_3d(positions)
    % Calculate total 3D path length
    path_length = 0;
    for i = 2:size(positions, 2)
        path_length = path_length + norm(positions(:, i) - positions(:, i-1));
    end
end