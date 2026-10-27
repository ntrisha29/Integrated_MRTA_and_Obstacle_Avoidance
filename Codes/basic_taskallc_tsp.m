% TSP
drone_pos = [0, 0, 0];
foi_pos = [6,7,11;
            13, 12,15;
            11.5,4.75,13;
            12,3.33,13.33;
            17,10,11];

% Combine start point with FOIs for the cost matrix
all_pts = [drone_pos; foi_pos];
n = size(all_pts, 1);

% 1. Create the Distance Matrix
dist_matrix = zeros(n, n);
for i = 1:n
    for j = 1:n
        dist_matrix(i,j) = norm(all_pts(i,:) - all_pts(j,:));
    end
end

% TSP using Permutations
drone_next_pos = 2:n;
perms_list = perms(drone_next_pos);
min_dist = inf;
best_path = [];

for i = 1:size(perms_list, 1)
    current_path = [1, perms_list(i, :), 1];
    % Distance
    d = 0;
    for k = 1:length(current_path)-1
        d = d + dist_matrix(current_path(k), current_path(k+1));
    end
    
    % Shortest distance
    if d < min_dist
        min_dist = d;
        best_path = current_path;
    end
end
disp(best_path);

fprintf('Shortest TSP Distance is %.4f meters\n', min_dist);

hold on;
grid on;

scatter3(all_pts(1,1), all_pts(1,2), all_pts(1,3), 150, 'g', 'filled', 'DisplayName', 'Start');
scatter3(all_pts(2:end,1), all_pts(2:end,2), all_pts(2:end,3), 120, 'r', '^', 'DisplayName', 'FOIs');

final_coords = all_pts(best_path, :);
plot3(final_coords(:,1), final_coords(:,2), final_coords(:,3), 'b-o', 'LineWidth', 2);
legend;
hold off;
% disp(perms_list);