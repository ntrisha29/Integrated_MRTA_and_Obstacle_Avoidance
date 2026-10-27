% TASK ALLOCATION

num_drones = 5;
num_FOIs = 5;
env_size = [20, 20, 20]; % Hangar dimensions

% try
%     rosinit('http://172.17.151.246:11311');
% catch
%     rosshutdown;
%     rosinit('http://172.17.151.246:11311');
% end
% 
% markerPub = rospublisher('/visualization_marker', 'visualization_msgs/Marker');
% pause(1);

w_dist = 1.0;  % Weight for Distance (1 meter = 1 cost unit)
w_battery = 1.0;  % Weight for Battery (1% drain = 1 cost unit)

% Initial Drone Positions
drones_pos = [0,0,0;
            20,0,0;
            20,20,0;
            0,20,0;
            0,1,0]; 

% Batter Percentage
drone_battery = [90;95;83;85;79];

% FOI Positions
foi_pos = [6,7,11;
            13,12,15;
            11.5,4.75,13;
            12,3.33,13.33;
            17,10,11];

CostMatrix = zeros(num_drones, num_FOIs);

for i = 1:num_drones
    for j = 1:num_FOIs
        dist = norm(drones_pos(i,:) - foi_pos(j,:));
        battery_percent = (100-drone_battery(i));
        CostMatrix(i, j) = (w_dist*dist)+(w_battery*battery_percent);
    end
end

% Drone Assignments
assignments = matchpairs(CostMatrix, 1000) 

total_cost = 0;
total_dist = 0;

for k = 1:size(assignments, 1)
    drone_idx = assignments(k, 1);
    foi_idx = assignments(k, 2);
    total_cost = total_cost + CostMatrix(drone_idx, foi_idx);
    total_dist = total_dist + norm(drones_pos(drone_idx,:) - foi_pos(foi_idx,:));
end

fprintf('Total Cost is %.4f meters\n',total_cost);

% % FOIs TO RVIZ
% for i = 1:num_FOIs
%     targetMsg = rosmessage(markerPub);
%     targetMsg.Header.FrameId = 'map';
%     targetMsg.Id = i;
%     targetMsg.Type = targetMsg.CUBE;
%     targetMsg.Action = targetMsg.ADD;
% 
%     targetMsg.Pose.Position.X = foi_pos(i,1);
%     targetMsg.Pose.Position.Y = foi_pos(i,2);
%     targetMsg.Pose.Position.Z = foi_pos(i,3);
%     targetMsg.Color.R = 1.0; targetMsg.Color.A = 1.0;
%     targetMsg.Scale.X = 0.5; targetMsg.Scale.Y = 0.5; targetMsg.Scale.Z = 0.5;
% 
%     send(markerPub, targetMsg);
% end

% Plot in MATLAB
figure;
hold on;
grid on;
axis equal;
xlabel('x'); ylabel('y'); zlabel('z');
view(45, 30);

scatter3(foi_pos(:,1), foi_pos(:,2), foi_pos(:,3),120, 'r', '^','filled',...
    DisplayName='FOIs');
scatter3(drones_pos(:,1), drones_pos(:,2), drones_pos(:,3),120, 'b', 'o','filled',...
    DisplayName='Agents');
legend;
% Connection links
for k = 1:size(assignments, 1)
    d_id = assignments(k, 1);
    t_id = assignments(k, 2);

    p1 = drones_pos(d_id, :);
    p2 = foi_pos(t_id, :);

    plot3([p1(1) p2(1)], [p1(2) p2(2)], [p1(3) p2(3)], 'k-', 'LineWidth', 1.5,'HandleVisibility', 'off');
end

% % Animation
% steps = 50;
% for step = 1:steps
%     for k = 1:size(assignments, 1)
%         drone_idx = assignments(k, 1);
%         f_idx = assignments(k, 2);
% 
%         current_pos = drones_pos(drone_idx,:) + (foi_pos(f_idx,:) - drones_pos(drone_idx,:)) * (step/steps);
% 
%         droneMsg = rosmessage(markerPub);
%         droneMsg.Header.FrameId = 'map';
%         droneMsg.Id = drone_idx + 10; % Unique ID different from FOIs
%         droneMsg.Type = droneMsg.SPHERE;
% 
%         droneMsg.Pose.Position.X = current_pos(1);
%         droneMsg.Pose.Position.Y = current_pos(2);
%         droneMsg.Pose.Position.Z = current_pos(3);
%         droneMsg.Color.B = 1.0; droneMsg.Color.A = 1.0;
%         droneMsg.Scale.X = 0.6; droneMsg.Scale.Y = 0.6; droneMsg.Scale.Z = 0.6;
% 
%         send(markerPub, droneMsg);
%     end
%     pause(0.1);
% end
% 
% % hold off;