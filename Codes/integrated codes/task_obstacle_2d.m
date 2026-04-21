%% AIRCRAFT HANGAR INSPECTION: TASK ALLOCATION + OBSTACLE AVOIDANCE
clc; clear; close all;

%% ==================== CONFIGURABLE PARAMETERS ====================
hangar_length = 50;
hangar_width  = 50;
hangar_height = 20;

% Task allocation weights
w_dist    = 1.0;
w_battery = 1.0;

% DWA / avoidance-related config
dwa_config.inflate         = 0.50;
dwa_config.goal_tol        = 0.18;
dwa_config.dt              = 0.10;
dwa_config.maxSteps        = 5000;
dwa_config.v_min           = 0.0;
dwa_config.v_max           = 1.2;
dwa_config.w_max           = 2.2;
dwa_config.a_max           = 2.5;
dwa_config.alpha_max       = 4.0;
dwa_config.sim_dt          = 0.06;
dwa_config.pred_time_open  = 1.6;
dwa_config.pred_time_tight = 1.0;
dwa_config.Nv_base         = 13;
dwa_config.Nw_base         = 31;
dwa_config.L_open          = 1.6;
dwa_config.L_tight         = 0.8;

%% ==================== LOAD AND SHIFT AIRCRAFT ====================
model = stlread('airplane.STL');
original_points = model.Points;

min_coords     = min(original_points);
shifted_points = original_points - min_coords;
offset         = [10, 10, 0];
shifted_points = shifted_points + offset;

%% ==================== FILTER FOR TOP SURFACE ====================
TR        = triangulation(model.ConnectivityList, shifted_points);
normals   = faceNormal(TR);
incenters = incenter(TR);

z_threshold       = 0.25 * max(shifted_points(:,3));
valid_mask        = (normals(:,3) > 0.2) & (incenters(:,3) > z_threshold);
valid_face_idx    = find(valid_mask);
valid_points      = incenters(valid_face_idx, :);

%% ==================== GENERATE FOIs ====================
y_min = min(valid_points(:,2));
y_max = max(valid_points(:,2));
y_center = (y_min + y_max) / 2;
fuselage_margin = (y_max - y_min) * 0.125;

left_wing_pts  = valid_points(valid_points(:,2) < y_center - fuselage_margin, :);
right_wing_pts = valid_points(valid_points(:,2) > y_center + fuselage_margin, :);
fuselage_pts   = valid_points(valid_points(:,2) >= y_center - fuselage_margin & ...
                              valid_points(:,2) <= y_center + fuselage_margin, :);

get_centroids = @(pts, k) kmeans(pts, k, 'Replicates', 3);

[~, C_left]  = get_centroids(left_wing_pts, 3);
[~, C_right] = get_centroids(right_wing_pts, 4);
[~, C_fuse]  = get_centroids(fuselage_pts, 4);

target_removal_zone = [20, 25, 6];
dist_to_target = sum((C_fuse - target_removal_zone).^2, 2);
[~, remove_idx] = min(dist_to_target);
C_fuse(remove_idx, :) = [];

tail_section_pts = valid_points(valid_points(:,1) < (min(valid_points(:,1)) + 5), :);
[~, left_tail_idx]  = min(tail_section_pts(:,2));
[~, right_tail_idx] = max(tail_section_pts(:,2));
C_empennage = [tail_section_pts(left_tail_idx, :); tail_section_pts(right_tail_idx, :)];

[~, max_x_idx] = max(valid_points(:,1));
nose_point = valid_points(max_x_idx, :);

all_targets = [C_left; C_right; C_fuse; C_empennage; nose_point];

final_fois = zeros(size(all_targets, 1), 3);
for i = 1:size(all_targets, 1)
    dists = sum((valid_points - all_targets(i,:)).^2, 2);
    [~, min_idx] = min(dists);
    final_fois(i,:) = valid_points(min_idx, :);
end

%% ==================== ADD OBSTACLES ====================
obstacles = addHangarObstacles(hangar_length, hangar_width);

%% ==================== DEFINE AGENTS ====================
num_drones = 5;
drones_pos = [ ...
     0,  0, 0;
    50,  0, 0;
     0, 50, 0;
    50, 50, 0;
    25,  0, 0];

drone_battery = [80; 80; 83; 85; 79];
colors = lines(num_drones);

%% ==================== CLUSTERING ====================
[foi_cluster_idx, cluster_centers] = kmeans(final_fois, num_drones, 'Start', drones_pos);

%% ==================== COST MATRIX (HUNGARIAN) ====================
CostMatrix = zeros(num_drones, num_drones);
for i = 1:num_drones
    for j = 1:num_drones
        dist = norm(drones_pos(i,:) - cluster_centers(j,:));
        battery_penalty = (100 - drone_battery(i));
        CostMatrix(i, j) = (w_dist * dist) + (w_battery * battery_penalty);
    end
end

assignments = matchpairs(CostMatrix, 1000);

%% ==================== CREATE OBSTACLE MAP ====================
grid_resolution = 0.2;
[X_grid, Y_grid] = meshgrid(0:grid_resolution:hangar_length, ...
                            0:grid_resolution:hangar_width);
obstacle_map = createObstacleMap(X_grid, Y_grid, obstacles, dwa_config.inflate);

%% ==================== PLAN PATHS WITH OBSTACLE AVOIDANCE ====================
fprintf('\n=== PLANNING PATHS WITH OBSTACLE AVOIDANCE ===\n');
total_system_flight_dist = 0;
total_assignment_cost    = 0;

all_drone_paths = cell(num_drones, 1);
all_drone_fois  = cell(num_drones, 1);

for k = 1:size(assignments, 1)
    d_id = assignments(k, 1);
    c_id = assignments(k, 2);

    task_cost = CostMatrix(d_id, c_id);
    total_assignment_cost = total_assignment_cost + task_cost;

    home_pos = drones_pos(d_id, :);
    assigned_fois = final_fois(foi_cluster_idx == c_id, :);

    dist_from_home = sqrt(sum((assigned_fois(:,1:2) - home_pos(1:2)).^2, 2));
    [~, sort_idx] = sort(dist_from_home);
    assigned_fois = assigned_fois(sort_idx, :);

    all_drone_fois{d_id} = assigned_fois;

    fprintf('Agent %d (Bat: %d%%) -> Cluster %d | FOIs: %d | Assign Cost: %.2f\n', ...
        d_id, drone_battery(d_id), c_id, size(assigned_fois, 1), task_cost);

    if isempty(assigned_fois)
        all_drone_paths{d_id} = home_pos;
        continue;
    end

    [optimal_order, min_dist] = tsp_with_obstacles(home_pos, assigned_fois, ...
        obstacles, dwa_config, obstacle_map, X_grid, Y_grid, grid_resolution);

    ordered_fois = assigned_fois(optimal_order, :);
    all_drone_paths{d_id} = build_obstacle_aware_path(home_pos, ordered_fois, ...
        obstacles, dwa_config, obstacle_map, X_grid, Y_grid, grid_resolution);

    total_system_flight_dist = total_system_flight_dist + min_dist;

    fprintf('  Flight Distance (with avoidance): %.2f m\n', min_dist);
end

fprintf('TOTAL SYSTEM ASSIGNMENT COST: %.2f\n', total_assignment_cost);
fprintf('TOTAL SYSTEM FLIGHT DISTANCE: %.2f m\n', total_system_flight_dist);

%% ==================== VISUALIZATION ====================
create_visualization(model, shifted_points, drones_pos, final_fois, ...
    foi_cluster_idx, assignments, all_drone_paths, all_drone_fois, ...
    obstacles, colors, hangar_length, hangar_width);

%% ==================== ANIMATE DRONE MOVEMENTS ====================
animate_drone_movements(all_drone_paths, drones_pos, final_fois, ...
    obstacles, colors, hangar_length, hangar_width, dwa_config);


%% ==================== HELPER FUNCTIONS ====================

function obstacles = addHangarObstacles(~, ~)
    % Better obstacle placement:
    % - not on top of bases
    % - not unnecessarily blocking obvious start/goal points
    % - still forces meaningful detours

    % Cylinders: [x, y, radius, height]
    obstacles.cylinders = [
        %14, 20, 2.2, 15;
        6, 36, 1.8, 10;
        %35, 18, 1.8, 12;
        40, 45, 2.0, 14;
        %11, 39, 1.4, 8
    ];

    % Squares: [x1, y1, x2, y2]
    obstacles.squares = [
        %45, 30, 19, 15;
        8, 10, 19, 15;
        %20, 5, 5, 5;
        41, 10, 46, 15
    ];

    % Circular approximation for DWA-style simplified checks
    obs_list = obstacles.cylinders(:,1:3);

    for i = 1:size(obstacles.squares, 1)
        x1 = obstacles.squares(i,1);
        y1 = obstacles.squares(i,2);
        x2 = obstacles.squares(i,3);
        y2 = obstacles.squares(i,4);

        center_x = (x1 + x2) / 2;
        center_y = (y1 + y2) / 2;
        radius = 0.5 * sqrt((x2-x1)^2 + (y2-y1)^2);

        obs_list = [obs_list; center_x, center_y, radius]; %#ok<AGROW>
    end

    obstacles.dwa_obs = obs_list;
end

function obstacle_map = createObstacleMap(X_grid, Y_grid, obstacles, inflate)
    obstacle_map = false(size(X_grid));

    % Cylinders
    for i = 1:size(obstacles.cylinders, 1)
        cx = obstacles.cylinders(i, 1);
        cy = obstacles.cylinders(i, 2);
        r  = obstacles.cylinders(i, 3) + inflate;

        dist = sqrt((X_grid - cx).^2 + (Y_grid - cy).^2);
        obstacle_map(dist <= r) = true;
    end

    % Squares
    for i = 1:size(obstacles.squares, 1)
        x1 = obstacles.squares(i, 1) - inflate;
        y1 = obstacles.squares(i, 2) - inflate;
        x2 = obstacles.squares(i, 3) + inflate;
        y2 = obstacles.squares(i, 4) + inflate;

        obstacle_map(X_grid >= x1 & X_grid <= x2 & ...
                     Y_grid >= y1 & Y_grid <= y2) = true;
    end
end

function [optimal_order, total_dist] = tsp_with_obstacles(start_pos, fois, ...
    obstacles, dwa_config, obstacle_map, X_grid, Y_grid, resolution)

    num_fois = size(fois, 1);

    if num_fois <= 1
        optimal_order = 1;
        if num_fois == 1
            total_dist = estimate_path_length(start_pos, fois(1,:), ...
                obstacles, dwa_config, obstacle_map, X_grid, Y_grid, resolution);
        else
            total_dist = 0;
        end
        return;
    end

    all_perms = perms(1:num_fois);
    min_dist = inf;
    optimal_order = 1:num_fois;

    all_points = [start_pos; fois];
    n_points = size(all_points, 1);
    dist_matrix = zeros(n_points);

    for i = 1:n_points
        for j = i+1:n_points
            d = estimate_path_length(all_points(i,:), all_points(j,:), ...
                obstacles, dwa_config, obstacle_map, X_grid, Y_grid, resolution);
            dist_matrix(i, j) = d;
            dist_matrix(j, i) = d;
        end
    end

    for p = 1:size(all_perms, 1)
        current_order = all_perms(p, :);

        route_dist = dist_matrix(1, current_order(1) + 1);

        for step = 1:num_fois-1
            route_dist = route_dist + dist_matrix(current_order(step) + 1, ...
                                                  current_order(step+1) + 1);
        end

        route_dist = route_dist + dist_matrix(current_order(end) + 1, 1);

        if route_dist < min_dist
            min_dist = route_dist;
            optimal_order = current_order;
        end
    end

    total_dist = min_dist;
end

function dist = estimate_path_length(p1, p2, obstacles, dwa_config, ...
    obstacle_map, X_grid, Y_grid, resolution)

    p1xy = move_point_out_of_obstacle(p1(1:2), obstacles, dwa_config.inflate);
    p2xy = move_point_out_of_obstacle(p2(1:2), obstacles, dwa_config.inflate);

    if is_line_clear(p1xy, p2xy, obstacles, dwa_config.inflate)
        dist = norm(p1xy - p2xy);
    else
        path = astar_grid(p1xy, p2xy, obstacle_map, X_grid, Y_grid, resolution);
        if ~isempty(path)
            dist = sum(sqrt(sum(diff(path).^2, 2)));
        else
            dist = 1.5 * (abs(p1xy(1) - p2xy(1)) + abs(p1xy(2) - p2xy(2)));
        end
    end
end

function tf = is_line_clear(p1, p2, obstacles, inflate)
    tf = true;
    num_samples = 60;

    for t = linspace(0, 1, num_samples)
        px = p1(1) + t * (p2(1) - p1(1));
        py = p1(2) + t * (p2(2) - p1(2));

        % Cylinders
        for i = 1:size(obstacles.cylinders, 1)
            cx = obstacles.cylinders(i, 1);
            cy = obstacles.cylinders(i, 2);
            r  = obstacles.cylinders(i, 3) + inflate;

            if (px - cx)^2 + (py - cy)^2 < r^2
                tf = false;
                return;
            end
        end

        % Squares
        for i = 1:size(obstacles.squares, 1)
            x1 = obstacles.squares(i, 1) - inflate;
            y1 = obstacles.squares(i, 2) - inflate;
            x2 = obstacles.squares(i, 3) + inflate;
            y2 = obstacles.squares(i, 4) + inflate;

            if px >= x1 && px <= x2 && py >= y1 && py <= y2
                tf = false;
                return;
            end
        end
    end
end

function path = astar_grid(start, goal, obstacle_map, X_grid, Y_grid, ~)
    [ny, nx] = size(obstacle_map);

    [~, start_x] = min(abs(X_grid(1,:) - start(1)));
    [~, start_y] = min(abs(Y_grid(:,1) - start(2)));
    [~, goal_x]  = min(abs(X_grid(1,:) - goal(1)));
    [~, goal_y]  = min(abs(Y_grid(:,1) - goal(2)));

    if obstacle_map(start_y, start_x) || obstacle_map(goal_y, goal_x)
        path = [];
        return;
    end

    open_set = [start_x, start_y, 0, heuristic(start_x, start_y, goal_x, goal_y)];
    closed_set = false(ny, nx);
    came_from = zeros(ny, nx, 2);
    g_score = inf(ny, nx);
    g_score(start_y, start_x) = 0;

    neighbors = [1 0; -1 0; 0 1; 0 -1; 1 1; 1 -1; -1 1; -1 -1];

    while ~isempty(open_set)
        [~, idx] = min(open_set(:,4));
        current = open_set(idx, 1:2);
        open_set(idx, :) = [];

        if closed_set(current(2), current(1))
            continue;
        end
        closed_set(current(2), current(1)) = true;

        if current(1) == goal_x && current(2) == goal_y
            path = reconstruct_path(came_from, X_grid, Y_grid, ...
                start_x, start_y, goal_x, goal_y);
            return;
        end

        for k = 1:size(neighbors, 1)
            nx2 = current(1) + neighbors(k, 1);
            ny2 = current(2) + neighbors(k, 2);

            if nx2 < 1 || nx2 > nx || ny2 < 1 || ny2 > ny
                continue;
            end
            if obstacle_map(ny2, nx2) || closed_set(ny2, nx2)
                continue;
            end

            cost = sqrt(neighbors(k,1)^2 + neighbors(k,2)^2);
            tentative_g = g_score(current(2), current(1)) + cost;

            if tentative_g < g_score(ny2, nx2)
                g_score(ny2, nx2) = tentative_g;
                came_from(ny2, nx2, :) = current;
                f = tentative_g + heuristic(nx2, ny2, goal_x, goal_y);
                open_set(end+1, :) = [nx2, ny2, tentative_g, f]; %#ok<AGROW>
            end
        end
    end

    path = [];
end

function h = heuristic(x, y, gx, gy)
    h = sqrt((gx - x)^2 + (gy - y)^2);
end

function path = reconstruct_path(came_from, X_grid, Y_grid, sx, sy, gx, gy)
    cells = [gx, gy];
    cx = gx; cy = gy;

    while ~(cx == sx && cy == sy)
        px = came_from(cy, cx, 1);
        py = came_from(cy, cx, 2);
        if px == 0 && py == 0
            path = [];
            return;
        end
        cells = [px, py; cells]; %#ok<AGROW>
        cx = px; cy = py;
    end

    path = zeros(size(cells, 1), 2);
    for i = 1:size(cells, 1)
        path(i, 1) = X_grid(1, cells(i, 1));
        path(i, 2) = Y_grid(cells(i, 2), 1);
    end
end

function p_out = move_point_out_of_obstacle(p_in, obstacles, inflate)
    p_out = p_in;

    for iter = 1:30
        moved = false;

        % Cylinders
        for i = 1:size(obstacles.cylinders, 1)
            cx = obstacles.cylinders(i,1);
            cy = obstacles.cylinders(i,2);
            r  = obstacles.cylinders(i,3) + inflate;

            v = p_out - [cx cy];
            d = norm(v);

            if d < r
                if d < 1e-6
                    v = [1 0];
                    d = 1;
                end
                p_out = [cx cy] + (r + 0.3) * (v / d);
                moved = true;
            end
        end

        % Squares
        for i = 1:size(obstacles.squares, 1)
            x1 = obstacles.squares(i,1) - inflate;
            y1 = obstacles.squares(i,2) - inflate;
            x2 = obstacles.squares(i,3) + inflate;
            y2 = obstacles.squares(i,4) + inflate;

            if p_out(1) >= x1 && p_out(1) <= x2 && p_out(2) >= y1 && p_out(2) <= y2
                dx_left   = abs(p_out(1) - x1);
                dx_right  = abs(x2 - p_out(1));
                dy_bottom = abs(p_out(2) - y1);
                dy_top    = abs(y2 - p_out(2));

                [~, idx] = min([dx_left dx_right dy_bottom dy_top]);

                switch idx
                    case 1
                        p_out(1) = x1 - 0.3;
                    case 2
                        p_out(1) = x2 + 0.3;
                    case 3
                        p_out(2) = y1 - 0.3;
                    case 4
                        p_out(2) = y2 + 0.3;
                end
                moved = true;
            end
        end

        if ~moved
            break;
        end
    end
end

function full_path = build_obstacle_aware_path(start_pos, ordered_fois, ...
    obstacles, dwa_config, obstacle_map, X_grid, Y_grid, resolution)

    full_path = start_pos;
    current_pos = start_pos;

    for i = 1:size(ordered_fois, 1)
        target = ordered_fois(i, :);

        p_start_xy = move_point_out_of_obstacle(current_pos(1:2), obstacles, dwa_config.inflate);
        p_goal_xy  = move_point_out_of_obstacle(target(1:2), obstacles, dwa_config.inflate);

        current_pos(1:2) = p_start_xy;
        target(1:2)      = p_goal_xy;

        if is_line_clear(current_pos(1:2), target(1:2), obstacles, dwa_config.inflate)
            num_interp = max(2, ceil(norm(current_pos(1:2) - target(1:2)) / 0.3));
            x_vals = linspace(current_pos(1), target(1), num_interp)';
            y_vals = linspace(current_pos(2), target(2), num_interp)';
            z_vals = linspace(current_pos(3), target(3), num_interp)';
            segment = [x_vals, y_vals, z_vals];
        else
            grid_path = astar_grid(current_pos(1:2), target(1:2), ...
                obstacle_map, X_grid, Y_grid, resolution);

            if ~isempty(grid_path)
    grid_path = smooth_grid_path(grid_path, obstacles, dwa_config.inflate);
    z_vals = linspace(current_pos(3), target(3), size(grid_path, 1))';
    segment = [grid_path, z_vals];
            else
                fprintf('Warning: No clear path found, using direct path\n');
                segment = [current_pos; target];
            end
        end

        if size(full_path, 1) > 0 && norm(full_path(end, :) - segment(1, :)) < 0.01
            full_path = [full_path; segment(2:end, :)]; %#ok<AGROW>
        else
            full_path = [full_path; segment]; %#ok<AGROW>
        end

        current_pos = target;
    end
function path2 = smooth_grid_path(path, obstacles, inflate)
    if size(path,1) <= 2
        path2 = path;
        return;
    end

    path2 = path;
    i = 1;
    while i < size(path2,1)-1
        j = i + 2;
        while j <= size(path2,1)
            if is_line_clear(path2(i,:), path2(j,:), obstacles, inflate)
                path2(i+1:j-1,:) = [];
                j = i + 2;
            else
                j = j + 1;
            end
        end
        i = i + 1;
    end
end
    p_start_xy = move_point_out_of_obstacle(current_pos(1:2), obstacles, dwa_config.inflate);
    p_goal_xy  = move_point_out_of_obstacle(start_pos(1:2), obstacles, dwa_config.inflate);

    current_pos(1:2) = p_start_xy;
    start_pos(1:2)   = p_goal_xy;

    if is_line_clear(current_pos(1:2), start_pos(1:2), obstacles, dwa_config.inflate)
        num_interp = max(2, ceil(norm(current_pos(1:2) - start_pos(1:2)) / 0.3));
        x_vals = linspace(current_pos(1), start_pos(1), num_interp)';
        y_vals = linspace(current_pos(2), start_pos(2), num_interp)';
        z_vals = linspace(current_pos(3), start_pos(3), num_interp)';
        return_segment = [x_vals, y_vals, z_vals];
    else
        grid_path = astar_grid(current_pos(1:2), start_pos(1:2), ...
            obstacle_map, X_grid, Y_grid, resolution);

        if ~isempty(grid_path)
    grid_path = smooth_grid_path(grid_path, obstacles, dwa_config.inflate);
    z_vals = linspace(current_pos(3), start_pos(3), size(grid_path, 1))';
    return_segment = [grid_path, z_vals];
        else
            return_segment = [current_pos; start_pos];
        end
    end

    full_path = [full_path; return_segment(2:end, :)];
end

function create_visualization(model, shifted_points, drones_pos, final_fois, ...
    foi_cluster_idx, assignments, all_drone_paths, ~, ...
    obstacles, colors, hangar_length, hangar_width)

    figure('Name', 'Task Allocation with Obstacle Avoidance', 'Position', [50, 50, 1200, 800]);

    %% Top view
    subplot(2, 2, [1, 3]);
    hold on; grid on; axis equal;
    xlim([0, hangar_length]); ylim([0, hangar_width]);
    xlabel('X (m)'); ylabel('Y (m)');
    title('Top View: Path Planning with Obstacle Avoidance');

    patch('Faces', model.ConnectivityList, 'Vertices', [shifted_points(:,1), ...
        shifted_points(:,2), zeros(size(shifted_points,1),1)], ...
        'FaceColor', [0.8 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.3, ...
        'HandleVisibility', 'off');

    % Cylinders
    for i = 1:size(obstacles.cylinders, 1)
        cx = obstacles.cylinders(i, 1);
        cy = obstacles.cylinders(i, 2);
        r  = obstacles.cylinders(i, 3);

        rectangle('Position', [cx-r, cy-r, 2*r, 2*r], ...
            'Curvature', [1, 1], 'FaceColor', [0.7 0.7 0.7], ...
            'EdgeColor', 'k', 'LineWidth', 1.5, 'HandleVisibility', 'off');
        text(cx, cy, 'Cyl', 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', 'FontSize', 9, 'FontWeight', 'bold', ...
    'BackgroundColor', 'w', 'Margin', 1);
    end

    % Squares
    for i = 1:size(obstacles.squares, 1)
        x1 = obstacles.squares(i,1);
        y1 = obstacles.squares(i,2);
        x2 = obstacles.squares(i,3);
        y2 = obstacles.squares(i,4);

        rectangle('Position', [x1, y1, x2-x1, y2-y1], ...
            'FaceColor', [0.6 0.6 0.6], 'EdgeColor', 'k', ...
            'LineWidth', 1.5, 'HandleVisibility', 'off');
        text((x1+x2)/2, (y1+y2)/2, 'Sq', 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', 'FontSize', 10, 'FontWeight', 'bold', ...
    'BackgroundColor', 'w', 'Margin', 1);
    end

    for i = 1:size(drones_pos, 1)
        scatter3(drones_pos(i,1), drones_pos(i,2), drones_pos(i,3), ...
            150, colors(i,:), 's', 'filled', ...
            'DisplayName', sprintf('Agent %d Base', i));
    end

    for i = 1:size(final_fois, 1)
        drone_id = find(arrayfun(@(x) any(assignments(:,2) == foi_cluster_idx(i) & ...
            assignments(:,1) == x), 1:size(drones_pos,1)), 1);

        if ~isempty(drone_id)
            scatter3(final_fois(i,1), final_fois(i,2), final_fois(i,3), ...
                80, colors(drone_id,:), '^', 'filled', 'HandleVisibility', 'off');
        end
    end

    for i = 1:length(all_drone_paths)
        if ~isempty(all_drone_paths{i})
            plot3(all_drone_paths{i}(:,1), all_drone_paths{i}(:,2), ...
                all_drone_paths{i}(:,3), 'Color', colors(i,:), ...
                'LineWidth', 2, 'DisplayName', sprintf('Agent %d Path', i));
        end
    end

    legend('Location', 'eastoutside');

    %% 3D view
    subplot(2, 2, [2, 4]);
    hold on; grid on;
    view(45, 30);
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    title('3D View with Obstacles');

    patch('Faces', model.ConnectivityList, 'Vertices', shifted_points, ...
        'FaceColor', [0.8 0.8 0.8], 'EdgeColor', 'none', ...
        'FaceAlpha', 0.5, 'HandleVisibility', 'off');

    for i = 1:size(obstacles.cylinders, 1)
        cx = obstacles.cylinders(i, 1);
        cy = obstacles.cylinders(i, 2);
        r  = obstacles.cylinders(i, 3);
        h  = obstacles.cylinders(i, 4);

        [Xc, Yc, Zc] = cylinder(r, 20);
        Xc = Xc + cx;
        Yc = Yc + cy;
        Zc = Zc * h;

        surf(Xc, Yc, Zc, 'FaceColor', [0.5 0.5 0.5], ...
            'EdgeColor', 'none', 'FaceAlpha', 0.7, 'HandleVisibility', 'off');
    end

    for i = 1:size(obstacles.squares, 1)
        x1 = obstacles.squares(i,1);
        y1 = obstacles.squares(i,2);
        x2 = obstacles.squares(i,3);
        y2 = obstacles.squares(i,4);
        h  = 8;

        vertices = [x1, y1, 0; x2, y1, 0; x2, y2, 0; x1, y2, 0;
                    x1, y1, h; x2, y1, h; x2, y2, h; x1, y2, h];
        faces = [1 2 3 4; 2 6 7 3; 6 5 8 7; 5 1 4 8; 1 5 6 2; 4 3 7 8];

        patch('Vertices', vertices, 'Faces', faces, ...
            'FaceColor', [0.5 0.5 0.5], 'EdgeColor', 'k', ...
            'FaceAlpha', 0.7, 'HandleVisibility', 'off');
    end

    for i = 1:length(all_drone_paths)
        if ~isempty(all_drone_paths{i})
            plot3(all_drone_paths{i}(:,1), all_drone_paths{i}(:,2), ...
                all_drone_paths{i}(:,3), 'Color', colors(i,:), ...
                'LineWidth', 2, 'HandleVisibility', 'off');
        end
    end

    scatter3(final_fois(:,1), final_fois(:,2), final_fois(:,3), ...
        50, 'r', 'filled', 'HandleVisibility', 'off');

    daspect([1 1 1]);
end



function animate_drone_movements(all_drone_paths, drones_pos, final_fois, ...
    obstacles, colors, hangar_length, hangar_width, dwa_config)

    figure('Name', 'Drone Movement Animation', 'Position', [100, 100, 1000, 800]);
    hold on; grid on; axis equal;
    xlim([0, hangar_length]); ylim([0, hangar_width]); zlim([0, 20]);
    view(45, 30);
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    title('Animated Drone Movement with Obstacle Avoidance');
    
    % Track inspected FOIs
    inspected_fois = false(size(final_fois, 1), 1);

    % Cylinders
    for i = 1:size(obstacles.cylinders, 1)
        cx = obstacles.cylinders(i,1);
        cy = obstacles.cylinders(i,2);
        r  = obstacles.cylinders(i,3);
        h  = obstacles.cylinders(i,4);

        [Xc, Yc, Zc] = cylinder(r, 20);
        Xc = Xc + cx;
        Yc = Yc + cy;
        Zc = Zc * h;

        surf(Xc, Yc, Zc, 'FaceColor', [0.7 0.7 0.7], ...
            'EdgeColor', 'none', 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
    end

    % Squares
    for i = 1:size(obstacles.squares, 1)
        x1 = obstacles.squares(i,1);
        y1 = obstacles.squares(i,2);
        x2 = obstacles.squares(i,3);
        y2 = obstacles.squares(i,4);
        h  = 8;

        vertices = [x1, y1, 0; x2, y1, 0; x2, y2, 0; x1, y2, 0;
                    x1, y1, h; x2, y1, h; x2, y2, h; x1, y2, h];
        faces = [1 2 3 4; 2 6 7 3; 6 5 8 7; 5 1 4 8; 1 5 6 2; 4 3 7 8];

        patch('Vertices', vertices, 'Faces', faces, ...
            'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'k', ...
            'FaceAlpha', 0.5, 'HandleVisibility', 'off');
    end

    scatter3(final_fois(:,1), final_fois(:,2), final_fois(:,3), ...
        100, 'r', '^', 'filled', 'DisplayName', 'FOIs');

    num_drones = length(all_drone_paths);
    drone_markers = gobjects(num_drones, 1);
    drone_trails = gobjects(num_drones, 1);
    trail_data = cell(num_drones, 1);

    for i = 1:num_drones
        if ~isempty(all_drone_paths{i}) && size(all_drone_paths{i}, 1) > 0
            drone_markers(i) = scatter3(drones_pos(i,1), drones_pos(i,2), ...
                drones_pos(i,3), 150, colors(i,:), 'o', 'filled', ...
                'DisplayName', sprintf('Drone %d', i));

            trail_data{i} = drones_pos(i, :);
            drone_trails(i) = plot3(NaN, NaN, NaN, 'Color', colors(i,:), ...
                'LineWidth', 1.5, 'HandleVisibility', 'off');
        end
    end

    max_len = 0;
    for i = 1:num_drones
        if ~isempty(all_drone_paths{i})
            max_len = max(max_len, size(all_drone_paths{i}, 1));
        end
    end

    for t = 1:max_len
        for i = 1:num_drones
            if ~isempty(all_drone_paths{i}) && t <= size(all_drone_paths{i}, 1)
                current_pos = all_drone_paths{i}(t, :);

                if isgraphics(drone_markers(i))
                    set(drone_markers(i), 'XData', current_pos(1), ...
                        'YData', current_pos(2), 'ZData', current_pos(3));
                end

                trail_data{i} = [trail_data{i}; current_pos];

                if isgraphics(drone_trails(i))
                    set(drone_trails(i), 'XData', trail_data{i}(:,1), ...
                        'YData', trail_data{i}(:,2), 'ZData', trail_data{i}(:,3));
                end
            end
        end

        for i = 1:num_drones
            if ~isempty(all_drone_paths{i}) && t <= size(all_drone_paths{i}, 1)
                current_pos = all_drone_paths{i}(t, :);

                for j = 1:size(final_fois, 1)
                    if ~inspected_fois(j) && norm(current_pos - final_fois(j, :)) < max(dwa_config.goal_tol, 0.25)
                        scatter3(final_fois(j,1), final_fois(j,2), ...
                            final_fois(j,3), 140, 'g', 'o', ...
                            'LineWidth', 2, 'HandleVisibility', 'off');
                        inspected_fois(j) = true;
                    end
                end
            end
        end

        drawnow;
        pause(0.5);
    end

    legend('Location', 'best');
    title('Animation Complete - All Drones Returned to Base');
end

   