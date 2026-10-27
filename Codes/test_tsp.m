% TSP FailSafe

start_pos = [0, 0, 0];
foi_pos = [6,7,11; 13,12,15; 11.5,4.75,13; 12,3.33,13.33; 17,10,11];
num_FOIs = size(foi_pos, 1);

battery_percent = 55;   %change
inspection_cost = 7;    %change
unvisited_foi = 1:num_FOIs;
current_pos = start_pos;
mission_cost = 0;
all_paths = {};
trip_idx = 1;
path = [start_pos];

while ~isempty(unvisited_foi)
    % nearest FOI
    dists = sqrt(sum((foi_pos(unvisited_foi,:) - current_pos).^2, 2));
    [min_d, idx] = min(dists);
    foi_idx = unvisited_foi(idx);
    target_pos = foi_pos(foi_idx, :);

    % cost to go to target and then return home
    dist_to_home = sqrt(sum((target_pos - start_pos).^2));

    total_cost = min_d + inspection_cost + dist_to_home;

    if battery_percent >= total_cost
        mission_cost = mission_cost + min_d + inspection_cost;
        battery_percent = battery_percent - (min_d + inspection_cost);
        current_pos = target_pos;
        path = [path; target_pos];
        unvisited_foi(idx) = [];
        fprintf('Visited FOI %d. Battery Percentage : %.1f \n', foi_idx, battery_percent);
    else
        dist_home = sqrt(sum((current_pos - start_pos).^2));
        mission_cost = mission_cost + dist_home;
        path = [path; start_pos];
        all_paths{trip_idx} = path;

        % Reset for next trip
        fprintf('\nReturn to home. Remaining FOIs: %d\n', length(unvisited_foi));
        battery_percent = 56; %change
        current_pos = start_pos;
        path = [start_pos];
        trip_idx = trip_idx + 1;
    end
end


% final trip
path = [path; start_pos];
all_paths{trip_idx} = path;

figure;
hold on;
grid on;
xlabel('x'); ylabel('y'); zlabel('z');
axis equal;
view(45,30);
colors = lines(length(all_paths));
for i = 1:length(all_paths)
    p = all_paths{i};
    plot3(p(:,1), p(:,2), p(:,3), '-o', 'Color', colors(i,:), 'LineWidth', 2, 'HandleVisibility', 'off');
end
scatter3(foi_pos(:,1), foi_pos(:,2), foi_pos(:,3), 120, 'r', '^','filled','DisplayName','FOIs');
legend;

fprintf('Total Cost is %.4f meters\n',mission_cost);